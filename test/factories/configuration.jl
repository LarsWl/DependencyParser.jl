using DependencyParser.DependencyParsing

# Configuration for "He wrote her a letter." after: SH, LA, RA, RA

function build_configuration()
  sentence = build_sentence()
  stack = Stack{Integer}()
  unshiftable = zeros(Bool, sentence.length)
  push!(stack, 0)
  push!(stack, 2)
  push!(stack, 3)

  buffer = Vector{Integer}([4, 5, 6])
  tree = build_uncomplete_tree()

  Configuration(buffer, stack, sentence, unshiftable, tree)
end