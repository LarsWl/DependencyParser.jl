export Layer;

struct Layer{F, M<:AbstractMatrix, B}
  weight::M
  bias::B
  σ::F
  function Dense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
    b = create_bias(W, bias, size(W,1))
    new{F,M,typeof(b)}(W, b, σ)
  end
end
