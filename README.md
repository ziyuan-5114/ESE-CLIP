# ESE-CLIP
This report studies how the ESE mechanism can improve CLIP-based image–text
retrieval under shallow- layer and low-dimensional settings. Standard CLIP is strong
when the final embedding is used, but its early layers and truncated prefixes are not
naturally reliable for retrieval. We therefore adapt the ESE idea to CLIP ViT-B/32[6]
and evaluate whether the first 256 dimensions, denoted as prefix_256, can become a
useful retrieval representation. 

The experiments show that Raw CLIP has almost no shallow-layer retrieval ability:
the L4–L6 average MeanR is only 1.31. After ESE training, the same metric increases
to 47.20, and L12 also improves from 34.54 to 68.20. Further layer-wise weight
tuning improves the shallow result to 54.26 with pw_l6_b4. A 64/192 feature split
then shows that task_192 carries most retrieval ability, while info_64 is better suited
for analysis, auxiliary constraints, and future PCA. The PCA experiments clarify an
important boundary: PCA can help or analyze info_64, but direct PCA rotation in the
final prefix_256 retrieval space reduces performance. The report concludes with a
future plan for Flickr30k and FAISS-based efficient retrieval. 

Keywords: CLIP; ESE; image–text retrieval; feature split; PCA; FAISS.
