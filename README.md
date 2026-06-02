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

# 1. Model and Dataset

The base model is CLIP ViT-B/32, implemented through openai/clip-vit-base- patch32. The dataset used in the current experiments is Flickr8k. Each image is
paired with captions, allowing both image-to-text and text-to-image retrieval evaluation. The main representation is prefix_256, the first 256 dimensions of the CLIP feature. In later
experiments, this 256-dimensional vector is split into info_64 and task_192. The split is
designed so that task_192 keeps the major retrieval function, while info_64 provides a
compact space for analysis and future PCA.

<img width="1717" height="1106" alt="image" src="https://github.com/user-attachments/assets/7833ca46-d5c8-4951-a8f9-116af954be21" />

The experiments are designed as a sequence. The first experiment tests whether ESE is
useful at all. The second experiment tunes layer-wise weights to improve the shallow
layers. The third experiment splits the representation into functional branches. The
fourth experiment tests PCA as a future direction and identifies its boundary.

<img width="1500" height="794" alt="image" src="https://github.com/user-attachments/assets/510f2b10-f703-461d-bb03-2cd4aa849490" />

# 2. Experiments and Results

Raw CLIP vs ESE-CLIP

<img width="1171" height="430" alt="image" src="https://github.com/user-attachments/assets/fc10e856-c3b4-4486-ac36-033aa29a77c2" />

Layer-wise Weight Tuning

<img width="1169" height="371" alt="image" src="https://github.com/user-attachments/assets/71de6072-7913-470b-8d4c-bea3f7ce7a0f" />


<img width="971" height="521" alt="image" src="https://github.com/user-attachments/assets/abfb8be1-f27d-49fb-8ce3-3fae510aea68" />

64/192 Feature Split

<img width="1027" height="552" alt="image" src="https://github.com/user-attachments/assets/15161fab-bfaf-49fb-b0f1-a7d0d87fcf44" />

PCA Boundary Analysis

<img width="1218" height="536" alt="image" src="https://github.com/user-attachments/assets/a4591cf7-e309-43d9-8aa5-953086cd191a" />

# 3. Conclusion

This report studies how ESE can improve CLIP-based image–text retrieval under shallowlayer and low- dimensional settings. The experiments show that Raw CLIP’s shallow
prefix_256 is almost unusable, but ESE-CLIP raises L4–L6 MeanR from 1.31 to 47.20. Layer-wise weight tuning further improves the shallow result to 54.26 with pw_l6_b4. The
64/192 split then shows that task_192 carries most retrieval ability, while info_64 is better
used for analysis and future PCA. 

The PCA experiments provide a practical boundary. PCA can help analyze or constrain
info_64, but it should not directly replace or rotate the final prefix_256 retrieval space. Overall, ESE is a useful mechanism for turning shallow and compressed CLIP
representations into practical retrieval features, and the project provides a clear path toward
larger-scale retrieval with Flickr30k and FAISS.









