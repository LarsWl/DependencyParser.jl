using DependencyParser.DependencyParsing

function build_updated_gold_state(config::Configuration)
  tree = build_gold_tree()

  GoldState(tree, config)
end