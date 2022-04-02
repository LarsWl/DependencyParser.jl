using DependencyParser
using Test
using DataStructures

# FACTORIES

include("factories/dependency_tree.jl")
include("factories/sentence.jl")
include("factories/configuration.jl")
include("factories/gold_state.jl")

@testset "DependencyParser.jl" begin
    include("dependecy_parsing/arc_eager/moves.jl")
    include("dependecy_parsing/arc_eager/gold_state.jl")
    include("dependecy_parsing/connlu.jl")
end
