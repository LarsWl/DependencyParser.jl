module DependencyParser
  using TextAnalysis
  using WordTokenizers
  
  include("core/core.jl")
  include("units.jl")
  include("dependency_parsing/dependency_parsing.jl")
  include("preprocess_units/preprocess_units.jl")
  include("pipeline.jl")
end
