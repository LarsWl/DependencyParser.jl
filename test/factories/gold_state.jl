using DependencyParser.DependencyParsing

function build_updated_gold_state(config::Configuration; system::ParsingSystem = ArcEagerSystem())
  tree = build_gold_tree()

  GoldState(tree, config, system)
end
