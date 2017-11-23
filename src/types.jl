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

function LinearPerceptron{T<:AbstractFloat}(X::AbstractArray{T},
                          alpha,
                          shuffle_epoch,
                          random_state,
                          max_epochs,
                          centralize)

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

#### Kernel Perceptron type
mutable struct KernelPerceptron{T<:AbstractFloat} <: PerceptronModel{T}
   X::AbstractArray{T}
   λ::Vector{T} # lagrange vector
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
   kernel::String
   width::T
   sv_x::Vector{T}
   sv_y::Vector{T}
end


function KernelPerceptron{T<:AbstractFloat}(X::AbstractArray{T},
                          alpha,
                          shuffle_epoch,
                          random_state,
                          max_epochs,
                          centralize,
                          kernel,
                          width)

   return KernelPerceptron(X,
                           zeros(T,size(X,1)),
                           alpha, # I will refactor to a constructor. Cleaner
                           rand(size(X,2)),
                           shuffle_epoch,
                           random_state,
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

## choosing types
######################################################################################################
function Model{T<:AbstractFloat}(X::AbstractArray{T},
               alpha,
               shuffle_epoch,
               random_state,
               max_epochs,
               centralize,
               kernel,
               width)

      if kernel == "linear"
         return LinearPerceptron(X,
                                 alpha,
                                 shuffle_epoch,
                                 random_state,
                                 max_epochs,
                                 centralize)
      elseif kernel == "rbf"
         return KernelPerceptron(X,
                                 alpha,
                                 shuffle_epoch,
                                 random_state,
                                 max_epochs,
                                 centralize,
                                 kernel,
                                 width)
      else
         error("Invalid Kernel name: $(kernel)")
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
