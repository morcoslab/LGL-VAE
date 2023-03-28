library('TreeDist')
library('permute')
library('ggplot2')

compare_trees <- function(tree1, tree2) {
  loaded_tree1 <- ape::read.tree(tree1)
  loaded_tree2 <- ape::read.tree(tree2)
  return( TreeDistance(loaded_tree1, loaded_tree2) )
}

compare_shuffled_tree <- function(tree) {
  loaded_tree <- ape::read.tree(tree)
  random_tree <- loaded_tree
  random_tree$tip.label = random_tree$tip.label[shuffle(random_tree$tip.label)]
  return( TreeDistance(loaded_tree, random_tree) )
}

real_list = list.files('subsampled_trees')
real_list =paste("subsampled_trees/",real_list,sep="")
real_list = real_list[shuffle(real_list)]

synth_list = list.files('generated_trees')
synth_list =paste("generated_trees/",synth_list,sep="")
synth_list = synth_list[shuffle(synth_list)]

desaturase_trees = paste('desaturase_trees/sampled_fasta_',
                         paste(seq(1,100),'.tree',sep=''),sep='')
globin_trees = paste('globin_trees/sampled_fasta_',
                         paste(seq(1,100),'.tree',sep=''),sep='')

## 100 random pairings of real trees and synth trees;
## 100 random pairings of real trees and real trees;
## 100 randomized real trees.
## Comparing phylogenies of the desaturase family and the globin family.
## 1-1 comparison; globin_tree_1 <-> desaturase_tree_1, etc. must be done.

real_synth_pairs = data.frame(real_list,synth_list)
real_real_pairs = data.frame(real_list, real_list[shuffle(real_list)] )
desaturase_globin_pairs = data.frame(desaturase_trees, globin_trees)

real_synth_results = mapply(compare_trees, real_synth_pairs$synth_list, 
                            real_synth_pairs$real_list)
real_real_results = mapply(compare_trees, real_real_pairs$real_list, 
                           real_real_pairs$real_list.shuffle.real_list..)
desat_globin_results = mapply(compare_trees, 
                              desaturase_globin_pairs$desaturase_trees,
                              desaturase_globin_pairs$globin_trees)
shuffled_real_results = sapply(real_list,compare_shuffled_tree)

# collect and plot results
results_df = data.frame(c(real_real_results,real_synth_results,
                          desat_globin_results,
                          shuffled_real_results),
                          rep(c("Real Sequences with Real Sequences",
                                "Real Sequences with Generated Sequences",
                                "Desaturase Sequences with Globin Sequences",
                              "Real Sequences with Shuffled Tree"), each=100)) 
colnames(results_df) = c("Data","Labels")

ggplot(data=results_df, aes(Data, fill=Labels)) +
  scale_color_brewer(palette='Set2') + 
  geom_histogram(bins=80,alpha=1,col=I('black')) + 
  facet_wrap(~Labels, ncol=1)+
  labs(x="Tree Comparison; Clustering Information Score",y="Count") + 
  theme(legend.position = "none")




