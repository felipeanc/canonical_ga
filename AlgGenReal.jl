###Libraries
using Random: bitrand, rand
using Distributions


#Canonical GA Struct
const Chromosome = Vector{Float64}
const Chromosomes = Vector{Chromosome}

##Individual representation
mutable struct Individual
  chromosome::Chromosome
  fitness::Float64

  function Individual(gen_num::Union{Int32,Int64}, fit=-Inf)
    ##Vector with gen_num elements in range [-gen_num, gen_num]
    c::Chromosome = sort(rand(Uniform(-gen_num, gen_num), gen_num), rev = true)
    new(c, fit)
  end

  Individual(c::Chromosome, apt=-Inf) = new(c, apt)
end

###Individual compare function
Base.isless(ind1::Individual, ind2::Individual) = ind1.fitness < ind2.fitness

###Multiple individuals
const Individuals = Vector{Individual}

##Population representation
struct Population
  individuals::Individuals
  size::Int

  ###Constructors
  function Population(size::Int32, gen_num::Int32)
    individuals = [Individual(gen_num) for _ = 1:size]
    new(individuals, size)
  end

  Population(individuals::Individuals) = new(individuals, length(individuals))
end

##Individual fitness evaluation
##using PERM FUNCTION 0, D, BETA, info below:
##https://www.sfu.ca/~ssurjano/perm0db.html
##Here we have a minimatizion problem. In order
##to achieve one maximization problem we use F(x) where:
##f(x) = PERM FUNCTION 0, D, BETA (min problem)
##F(x) = 1/(1+f(x)) (max problem)
function evaluate(ind::Individual)
  xx::Chromosome = ind.chromosome
  b = 100
  d = length(xx)

  ii = [x for x = 1:d]
  jj = mapreduce(permutedims, vcat, [ii for _ in 1:d])
  xxmat = mapreduce(permutedims, vcat, [xx for _ in 1:d])

  inner = (jj .+ b) .* (xxmat .^ ii .- (1 ./ jj) .^ ii)
  inner = sum(inner, dims = 2)
  ind.fitness = abs(sum(inner .^ 2))
  ind.fitness = 1/(1 + ind.fitness) #Minimization to maximization problem
end

##Population fitness evaluation
function evaluate(pop::Population)
  for ind in pop.individuals
    evaluate(ind)
  end
  sort!(pop.individuals)
end

struct GA
  pop_size::Int32
  dimension::Int32
  cross_prob::Float64
  mut_prob::Float64
  generation_limit::Int32
end

##Roulette wheel selection
function roulette_selection(pop::Population)
  individuals::Individuals = pop.individuals
  total_fitness = sum(ind -> ind.fitness, individuals)
  function select()::Int
    prob = rand() * total_fitness
    psum = 0.0
    for i = pop.size:-1:1
      psum += individuals[i].fitness
      if prob <= psum
        return i
      end
    end
  end

  return [ copy(individuals[select()].chromosome) for _ = 1:pop.size ]
end

##Tournament selection
function tournament_selection(pop::Population, k::Int = 3)
  individuals::Individuals = pop.individuals
  scores = [i for i = 1:pop.size]
  # println("Pop: ")
  # print_pop(pop)
  function select()::Int
    tournament = sample(scores, k; replace = false)
    i = maximum(tournament)
    return i
  end

  return [ copy(individuals[select()].chromosome) for _ = 1:pop.size ]
end

const Coin() = rand() < 0.5

##Uniform crossover
function uniform_crossover(cs::Chromosomes, ga::GA)
  for i = 1:2:ga.pop_size
    if rand() < ga.cross_prob
      p1 = cs[i] #Heads
      p2 = cs[i+1] #Tails
      for j = 1:ga.dimension
        if Coin() #Tails
          p1[j], p2[j] = p2[j], p1[j]
        end
      end
    end
  end
end

function singlepoint_crossover(cs::Chromosomes, ga::GA, crosspoint=-Inf)
  if crosspoint == -Inf
    crosspoint = rand(1:ga.dimension-1)
  end

  for i = 1:2:ga.pop_size
    if rand() < ga.cross_prob
     p1 = cs[i]
     p2 = cs[i+1]
     ofsp1 = [p1[1:crosspoint];p2[crosspoint+1:length(p2)]]
     ofsp2 = [p2[1:crosspoint];p1[crosspoint+1:length(p1)]]
     p1 = ofsp1
     p2 = ofsp2
    end
  end
end


function twopoint_crossover(cs::Chromosomes, ga::GA, l = -Inf, r = -Inf)
  if l == -Inf
    l = rand(1:length(cs[1])-1)
  end
  if r == -Inf
    r = rand(l:length(cs[1])-1)
  end

  for i = 1:2:ga.pop_size
    if rand() < ga.cross_prob
     p1 = cs[i]
     p2 = cs[i+1]
     p1[l:r], p2[l:r] = p2[l:r], p1[l:r]
    end
  end
end

function npoint_crossover(cs::Chromosomes, ga::GA, n_points = 2)

  function singlepoint_crossover(cs1::Chromosome, cs2::Chromosome)
    crosspoint = rand(1:length(cs1)-1)
    cs1 = [cs1[1:crosspoint];cs2[crosspoint+1:length(cs2)]]
    cs2 = [cs2[1:crosspoint];cs1[crosspoint+1:length(cs1)]]
  end

  function twopoint_crossover(cs1::Chromosome, cs2::Chromosome)
    l = rand(1:length(cs1)-1)
    r = rand(l:length(cs1)-1)
    cs1[l:r], cs2[l:r] = cs2[l:r], cs1[l:r]
  end

  for i = 1:2:ga.pop_size
    if rand() < ga.cross_prob
      p1 = cs[i]
      p2 = cs[i+1]
      if n_points == 1
        singlepoint_crossover(p1, p2)
      end
      if n_points == 2
        twopoint_crossover(p1, p2)
      end

      if n_points > 2
        points = sample([i for i = 1:ga.dimension], n_points; replace = false)
        sort!(points)
        global cross = false
        global l = 1
        for j = 1:n_points
          r = points[j]
          global cross = !cross
          global l
          if cross
            p1[l:r], p2[l:r] = p2[l:r], p1[l:r]
          end
          l = r
        end
      end
    end
  end
end

function arithmetic_crossover(cs::Chromosomes, ga::GA, alpha = 0.5)
  for i = 1:2:ga.pop_size
    if rand() < ga.cross_prob
     p1 = cs[i]
     p2 = cs[i+1]
     k = rand(1:ga.dimension)
     p1[k] = alpha * p2[k] + ((1 - alpha) * p1[k])
     p2[k] = alpha * p1[k] + ((1 - alpha) * p2[k])
    end
  end
end

function simple_arithmetic_crossover(cs::Chromosomes, ga::GA, alpha = 0.5)
  for i = 1:2:ga.pop_size
    if rand() < ga.cross_prob
     p1 = cs[i]
     p2 = cs[i+1]
     k = rand(1:ga.dimension-1)
     for j = 1:k
      p1[j] = alpha * p2[j] + ((1 - alpha) * p1[j])
      p2[j] = alpha * p1[j] + ((1 - alpha) * p2[j])
     end
    end
  end
end

function whole_arithmetic_crossover(cs::Chromosomes, ga::GA, alpha = 0.5)
  for i = 1:2:ga.pop_size
    if rand() < ga.cross_prob
     p1 = cs[i]
     p2 = cs[i+1]
     for j = 1:ga.dimension
      p1[j] = alpha * p2[j] + ((1 - alpha) * p1[j])
      p2[j] = alpha * p1[j] + ((1 - alpha) * p2[j])
     end
    end
  end
end

function mutation(cs::Chromosomes, ga::GA)
	for i=1:ga.pop_size, j=1:ga.dimension
        if rand() < ga.mut_prob
            cs[i][j] = -cs[i][j]
        end
    end
end

function uniform_mutation(cs::Chromosomes, ga::GA)
  for i=1:ga.pop_size, j=1:ga.dimension
    if rand() < ga.mut_prob
      cs[i][j] = rand(Uniform(-ga.dimension, ga.dimension))
    end
  end 
end

function stats(pop::Population)
  stats = zeros(Float64, 3)

  worst = pop.individuals[1].fitness
  best = pop.individuals[length(pop.individuals)].fitness
  psum = sum(x -> x.fitness, pop.individuals)
  avg = psum/pop.size

  stats[1] = best
  stats[2] = worst
  stats[3] = avg
  return stats
end

function print_pop(pop::Population)
  println()
  for ind in pop.individuals
    println(ind)
  end
end

function exec(ga::GA)::Population
  pop = Population(ga.pop_size, ga.dimension)
  evaluate(pop)
  s = stats(pop)
  println("Gen.: 0 → best: ", s[1], ", worst: ", s[2], ", avg.: ", s[3])
  for g = 1:ga.generation_limit
    if g % 100 == 0
      println("Gen.: $g → best: ", s[1], ", worst: ", s[2], ", avg.: ", s[3])
      
    end
    selected = roulette_selection(pop)
    uniform_crossover(selected, ga)
    uniform_mutation(selected, ga)
    pop = Population([Individual(i) for i in selected])
    evaluate(pop)
    s = stats(pop)
    println("Gen.: $g → best: ", s[1], ", worst: ", s[2], ", avg.: ", s[3])
  end
  pop
end

const revFit(x) = (1-x)/x

function main()
  dimension::Int32 = 2
  pop_size::Int32 = 20
  cross_prob::Float64 = 0.9
  mut_prob::Float64 = 0.08
  gen_limit::Int32 = 1000
  iterations::Int32 = 30
  ga = GA(pop_size, dimension, cross_prob, mut_prob, gen_limit)

  best = Individuals(undef, iterations)
  for i in 1:iterations
    println("\n########################## Iteration $i ##########################")
    pop::Population = exec(ga)
    individuals = sort(pop.individuals, rev=true)
    best[i] = individuals[1]
  end
  best = sort(best, rev=true)
  expected = [1/i for i = 1:dimension]
  println("Optimum: 0, $expected", 
          "\nBest: ", revFit(best[1].fitness), ", ", best[1].chromosome, 
          "\nWorst: ", revFit(best[iterations].fitness), ", ", best[iterations].chromosome)
end

main()
