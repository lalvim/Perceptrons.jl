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
