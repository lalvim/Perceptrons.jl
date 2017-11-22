module Perceptrons

include("utils.jl")
include("types.jl")
include("linear_perceptron.jl")



"""
    fit(X::Matrix{:<AbstractFloat},Y::AbstractArray{:<AbstractFloat}; copydata::Bool=true, centralize::Bool=true, kernel="", width=1.0)

Perceptron algorithm.

# Arguments
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
- `centralize::Bool = true`: If you want to z-score columns. Recommended if not z-scored yet.
- `kernel::AbstractString = "rbf"`: If you want to apply a nonlinear Perceptron with gaussian Kernel.
- `width::AbstractFloat = 1.0`: Rbf Kernel width (Only if kernel="rbf").
"""
function fit{T<:AbstractFloat}(X::AbstractArray{T},
                               Y::AbstractArray{T};
                               copydata::Bool         = true,
                               centralize::Bool       = true,
                               kernel                 = "linear",
                               width                  = 1.0,
                               alpha                  = 1.0e-2,
                               shuffle_epoch          = false,
                               random_state           = 42,
                               max_epochs             = 5,
                               centralize             = true
                               )
    X = X[:,:]
    check_constant_cols(X)
    check_constant_cols(Y)

    check_params(kernel)

    check_data(X, Y)

    Xi =  (copydata ? deepcopy(X) : X)
    Yi =  (copydata ? deepcopy(Y) : Y)

    model = LinearPerceptron(alpha,
                             Array{Real}(1),
                             shuffle_epoch,
                             random_state,
                             max_epochs,
                             0,
                             Array{Integer}(1),
                             mean(X,1),
                             std(X,1),
                             centralize,
                             size(X,2)
                             )

    Xi =  (centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)
    model.centralize  = (centralize ? true: false)

    trainer(model,Xi,Yi)

    return model
end


"""
    predict(model::Perceptron.Model; X::AbstractArray{:<AbstractFloat}; copydata::Bool=true)

A Perceptron predictor.

# Arguments
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
"""
function predict{T<:AbstractFloat}(model::PerceptronModel{T},
                                    X::AbstractArray{T};
                                    copydata::Bool=true)

    X = X[:,:]
    check_data(X,model.nfeatures)

    Xi =  (copydata ? deepcopy(X) : X)
    Xi =  (model.centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)

    Yi =  predictor(model,Xi)

    return Yi
end


end
