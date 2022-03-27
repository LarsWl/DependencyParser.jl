using DependencyParser.DependencyParsing

# Build gold tree for sentence: "He wrote her a letter." 
function build_gold_tree()
  root = TreeNode(0)
  nodes = vec(
    [
      TreeNode(1, "SBJ", 2),
      TreeNode(2, "PRD", 0),
      TreeNode(3, "IOBJ", 2),
      TreeNode(4, "DET", 5),
      TreeNode(5, "DOBJ", 2),
      TreeNode(6, "P", 2)
    ]
  )
  length = 6

  DependencyTree(root, nodes, length)
end

# Tree for "He wrote her a letter." after: SH, LA, RA, RA
function build_uncomplete_tree()
  root = TreeNode(0)
  nodes = vec(
    [
      TreeNode(1, "SBJ", 2),
      TreeNode(2, "PRD", 0),
      TreeNode(3, "IOBJ", 2),
      TreeNode(4, EMPTY_LABEL, EMPTY_NODE),
      TreeNode(5, EMPTY_LABEL, EMPTY_NODE),
      TreeNode(6, EMPTY_LABEL, EMPTY_NODE)
    ]
  )
  length = 6

  DependencyTree(root, nodes, length)
end