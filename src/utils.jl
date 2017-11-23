

## Auxiliary functions

## checks PLS input data and params
function check_data{T<:AbstractFloat}(X::Matrix{T},Y::Union{Vector{T},Matrix{T}})
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    !isempty(Y) ||
        throw(DimensionMismatch("Empty target data (Y)."))
    size(X, 1) == size(Y, 1) ||
        throw(DimensionMismatch("Incompatible number of rows of input data (X) and target data (Y)."))
end

function check_data{T<:AbstractFloat}(X::Matrix{T},nfeatures::Int)
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    size(X, 2) == nfeatures ||
        throw(DimensionMismatch("Incompatible number of columns of input data (X) and original training X columns."))
end


function check_params(kernel::AbstractString)
    kernel == "rbf" || kernel == "linear" || error("kernel must be kernel='linear' or kernel='rbf'")
end

## checks constant columns
check_constant_cols{T<:AbstractFloat}(X::Matrix{T}) = size(X,1)==1 || (i=find(all(X .== X[1,:]',1))) == [] || error("You must remove constant columns $i of input data (X) before train")
check_constant_cols{T<:AbstractFloat}(Y::Vector{T}) = length(Y)==1 || length(unique(Y)) != 1 || error("Your target values are constant. All values are equal to $(Y[1])")

## Preprocessing data using z-score statistics. this is due to the fact that if X and Y are z-scored, than X'Y returns for W vector a pearson correlation for each element! :)
centralize_data{T<:AbstractFloat}(D::AbstractArray{T}, m::AbstractArray{T}, s::AbstractArray{T})   = (D .-m)./s
centralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                   = (D -m)/s

decentralize_data{T<:AbstractFloat}(D::AbstractArray{T}, m::AbstractArray{T}, s::AbstractArray{T}) = D .*s .+m
decentralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                 = D *s +m

check_linear_binary_labels{T<:Number}(Y::Vector{T}) = length(setdiff(Y,[0.,1.]))==0 || error("Your target values must be 1 and 0 only. You have $(unique(Y)) distinct labels")
