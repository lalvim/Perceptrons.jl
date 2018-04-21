#### Constants
const MODEL_FILENAME = "perceptron_model.jld" # jld filename for storing the model
const MODEL_ID       = "perceptron_model"     # if od the model in the filesystem jld data

#### An abstract perceptron model
abstract type PerceptronModel{T} end

#### Linear Perceptron type
mutable struct LinearPerceptron{T<:AbstractFloat} <: PerceptronModel{T}
   α::T
   Θ::Vector{T}
   shuffle_epoch::Bool
   random_state::Integer
   max_epochs::Integer
   last_epoch::Integer
   history::Vector{Integer}
   mx::Matrix{T}          # mean stat after for z-scoring input data (X)
   sx::Matrix{T}          # standard deviation stat after for z-scoring target data (X)
   centralize::Bool
   nfeatures::Integer
end

function LinearPerceptron(X::AbstractArray{T},
                          alpha,
                          shuffle_epoch,
                          random_state,
                          max_epochs,
                          centralize) where T<:AbstractFloat

   return LinearPerceptron(alpha, # I will refactor to a constructor. Cleaner
                           Vector{T}(1),
                           shuffle_epoch,
                           random_state,
                           max_epochs,
                           0,
                           Vector{Integer}(1),
                           mean(X,1),
                           std(X,1),
                           centralize,
                           size(X,2))

end
####################################################################################

#### Linear Perceptron type
mutable struct VotedPerceptron{T<:AbstractFloat} <: PerceptronModel{T}
   α::T
   Θ#::Dict{Integer,Vector{T}}
   c#::Dict{Integer,Integer}
   k::Integer
   shuffle_epoch::Bool
   random_state::Integer
   max_epochs::Integer
   last_epoch::Integer
   history::Vector{Integer}
   mx::Matrix{T}          # mean stat after for z-scoring input data (X)
   sx::Matrix{T}          # standard deviation stat after for z-scoring target data (X)
   centralize::Bool
   nfeatures::Integer
end

function VotedPerceptron(X::AbstractArray{T},
                          alpha,
                          shuffle_epoch,
                          random_state,
                          max_epochs,
                          centralize) where T<:AbstractFloat

   return VotedPerceptron(alpha, # I will refactor to a constructor. Cleaner
                           nothing,
                           nothing,
                           0,
                           shuffle_epoch,
                           random_state,
                           max_epochs,
                           0,
                           Vector{Integer}(1),
                           mean(X,1),
                           std(X,1),
                           centralize,
                           size(X,2))

end


####################################################################################

#### Kernel Perceptron type
mutable struct KernelPerceptron{T<:AbstractFloat} <: PerceptronModel{T}
   λ::Vector{T} # lagrange vector
   max_epochs::Integer
   last_epoch::Integer
   history::Vector{Integer}
   mx::Matrix{T}          # mean stat after for z-scoring input data (X)
   sx::Matrix{T}          # standard deviation stat after for z-scoring target data (X)
   centralize::Bool
   nfeatures::Integer
   kernel::String
   width::T
   sv_x::Vector{T}
   sv_y::Vector{T}
end


function KernelPerceptron(X::AbstractArray{T},
                          max_epochs,
                          centralize,
                          kernel,
                          width) where T<:AbstractFloat

   return KernelPerceptron(zeros(T,size(X,1)),
                           max_epochs,
                           0,
                           Vector{Integer}(1),
                           mean(X,1),
                           std(X,1),
                           centralize,
                           size(X,2),
                           kernel,
                           width,
                           Vector{T}(1),
                           Vector{T}(1))

end

####################################################################################

#### Averaged Perceptron type
mutable struct AveragedPerceptron{T<:AbstractFloat} <: PerceptronModel{T}
   α::T
   Θ
   shuffle_epoch::Bool
   random_state::Integer
   max_epochs::Integer
   last_epoch::Integer
   history::Vector{Integer}
   mx::Matrix{T}          # mean stat after for z-scoring input data (X)
   sx::Matrix{T}          # standard deviation stat after for z-scoring target data (X)
   centralize::Bool
   nfeatures::Integer
end

function AveragedPerceptron(X::AbstractArray{T},
                          alpha,
                          shuffle_epoch,
                          random_state,
                          max_epochs,
                          centralize) where T<:AbstractFloat

   return AveragedPerceptron(alpha, # I will refactor to a constructor. Cleaner
                           0,
                           shuffle_epoch,
                           random_state,
                           max_epochs,
                           0,
                           Vector{Integer}(1),
                           mean(X,1),
                           std(X,1),
                           centralize,
                           size(X,2))

end





## choosing types
######################################################################################################
function Model(X::AbstractArray{T},
               alpha,
               shuffle_epoch,
               random_state,
               max_epochs,
               centralize,
               kernel,
               width,
               mode) where T<:AbstractFloat

      if mode == "linear"
         return LinearPerceptron(X,
                                 alpha,
                                 shuffle_epoch,
                                 random_state,
                                 max_epochs,
                                 centralize)
      elseif mode == "kernel"
         return KernelPerceptron(X,
                                 max_epochs,
                                 centralize,
                                 kernel,
                                 width)
      elseif mode == "voted"
      return VotedPerceptron(X,
                           alpha,
                           shuffle_epoch,
                           random_state,
                           max_epochs,
                           centralize)
      elseif mode == "averaged"
      return AveragedPerceptron(X,
                           alpha,
                           shuffle_epoch,
                           random_state,
                           max_epochs,
                           centralize)
      else
         error("Invalid perceptron mode name: $(mode). \n Cadidates are: linear, kernel, voted or averaged")
      end
end

######################################################################################################
## Load and Store models (good for production)
function load(; filename::AbstractString = MODEL_FILENAME, modelname::AbstractString = MODEL_ID)
    local M
    jldopen(filename, "r") do file
        M = read(file, modelname)
    end
    M
end

function save(M::PerceptronModel; filename::AbstractString = MODEL_FILENAME, modelname::AbstractString = MODEL_ID)
    jldopen(filename, "w") do file
        write(file, modelname, M)
    end
end
