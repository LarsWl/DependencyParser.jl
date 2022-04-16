export Sequence
export next!

mutable struct Sequence
  id::Int32

  Sequence() = new(0)
end

function next!(seq::Sequence)
  seq.id += 1
end