
# A gaussian kernel function
@inline function Φ{T<:AbstractFloat}(x::Vector{T},
                                     y::Vector{T},
                                     r::T=1.0)
    n  = 1.0 / sqrt(2π*r)
    s  = 1.0 / (2r^2)
    return n*exp(-s*sum((x.-y).^2))
end

# A kernel matrix
function ΦΦ{T<:AbstractFloat}(X::AbstractArray{T},
                              r::T=1.0)
    n = size(X,1)
    K = zeros(n,n)
    for i=1:n
        for j=1:i
            K[i, j] = Φ(X[i, :], X[j, :],r)
            K[j, i] = K[i, j]
        end
        K[i, i] = Φ(X[i, :], X[i, :],r)
    end
    K
end

# A kernel matrix for test data
function ΦΦ{T<:AbstractFloat}(X::AbstractArray{T},
                              Z::AbstractArray{T},
                              r::T=1.0)
    (nx,mx)    = size(X)
    (nz,mz)    = size(Z)
    K          = zeros(T,nz, nx)
    for i=1:nz
        for j=1:nx
            K[i, j] = Φ(Z[i, :], X[j, :],r)
        end
    end
    K
end

@inline function ∑(λ,y,n,K)
    #    sum = .0
    #    for i=1:n
    #        sum += λ[i]*y[i]*K[i,j]
    #    end
    #    return sum
    return sum(λ .* y .* K)
end

@inline function sign(val)
    return  (val >=0 ? 1.0: -1.0 )
end

function trainer{T<:AbstractFloat}(model::KernelPerceptron{T},
	                              X::AbstractArray{T},
        						  Y::Vector{T})
   Y[Y .== 0]  = -1 # fix in the future outside this function
   max_epochs  = model.max_epochs
   λ           = model.λ    # langrange multipliers
   K           = ΦΦ(X,model.width) # computing the kernel gram matrix
   n           = size(X,1)
   history     = []
   nerrors     = Inf
   epochs      = 0
   while  nerrors>0 && epochs < max_epochs
   # stops when error is equal to zero or grater than last_error or reached max iterations
       nerrors = 0
       # weight updates for all samples
       for i=1:n
          yp   = sign(∑(λ,Y,n,K[:,i]))
          if Y[i] != yp
             nerrors +=1
			 λ[i]    += 1    # missclassification counter for sample i
          end
		 end
       epochs+=1
       push!(history,nerrors)
   end
   if nerrors > 0
      warn("[Kernel Perceptron] Train not converged. Max epochs $(max_epochs) reached. Error history: $(history) \n Try to increase max_epochs or change kernel params.")
   end

   # storing only the tough samples ("support vectors")
   sv               = λ .> 0
   model.λ          = λ[sv]
   model.sv_x       = vec(X[sv,:])
   model.sv_y       = Y[sv]
   model.history    = history
   model.last_epoch = epochs

   println("[Kernel perceptron] #$(length(model.λ)) support vectors out of $(n) samples.")


end

function predictor{T<:AbstractFloat}(model::KernelPerceptron{T},
	                                    X::AbstractArray{T})

   width        = model.width
   sv_x,sv_y,λ  = model.sv_x,model.sv_y,model.λ

   k   = size(sv_y,1)
   n   = size(X,1)
   y   = zeros(T,n)
   for i=1:n
      s = .0
      for j=1:k # can be vectorized in the future.
          s += λ[j] * sv_y[j] * Φ(X[i,:],sv_x[j,:],width) # this is simply a weighted voting into a kernel space
      end
      y[i] = s
   end
   y   = sign.(y)
   y[y .== -1]  = 0 # fix in the future outside this function!!
   return y

end
