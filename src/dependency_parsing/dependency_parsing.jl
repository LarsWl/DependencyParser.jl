module DependencyParsing
  using ..DependencyParser.Core
  
  include("settings.jl")
  include("parsing_system.jl")
  include("transition.jl")
  include("tree_node.jl")
  include("dependency_tree.jl")
  include("connlu.jl")
  include("configuration.jl")
  include("arc_eager/arc_eager.jl")
  include("training_context.jl")
  include("model.jl")
  include("dep_parser.jl")
end