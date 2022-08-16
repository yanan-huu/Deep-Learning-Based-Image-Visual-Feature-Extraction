## Master Thesis Codebase for “Search by Image: Deep Learning Based Image Visual Feature Extraction”
## Abstrct
In recent years, the expansion of the Internet has brought an explosion of visual information, including social media, medical photographs, and digital history. This massive amount of visual content generation and sharing presents new challenges, especially when searching for similar information in databases —— Content-Based Image Retrieval (CBIR). Feature extraction is the foundation of image retrieval, making research into obtaining concrete features and representations of image content a vital concern. 

In feature extraction module, We first pre-process the target image and input it into a CNN to obtain feature maps for different channels. These feature maps can be aggregated into compact and global uniform descriptors by pooling. Then these global descriptors are further dimensionalised and normalised by whitening methods to obtain image feature vectors that are easy to compute and compare. In this process, the accuracy of the retrieval depends on how accurately the final feature vectors represent the meaning expressed by the target image. Therefore various CNN network structures, pooling methods and whitening methods are proposed to get more concrete feature vectors.

In this thesis, our study (1) fine tunes the pre-trained CNNs,  (2) optimizes the application of second order attention information in feature map, (3) applies and compares popular feature enhancement methods in both aggregating and whitening, (4) explores how to combine all strengths, and (5) propose a new model \textit{ResNet-SOI}, which achieves 53.4(M) and 59.2(M) mAP on the challenging benchmark \textit{ROxford5k+1M} and\textit{ RParis6k+1M}, and outperforms the state-of-art methods. Our prototype with GUI is available on GitHub(\url{https://github.com/yanan-huu/Image-Search-Engine-for-Historical-Research}).
## Our pipeline
![pipeline](https://user-images.githubusercontent.com/76591676/181504716-76a20f35-3485-4489-8f81-1104651e2c05.png)

## Usage
<details><summary><b>Training</b></summary>

<p>

We have already trained the model "Resnet101-solar-best" with good results, which is stored at https://drive.google.com/drive/folders/1JbGNvQgqKm7GiUvOqw1DSncSVR3k0xbm?usp=sharing. We recommend that you use ths pre-trained model. If you want to use our pre-trined model, download it and place it in ~/data/networks/ , then skip the following instructions directly to next part.

If you wish to retrain the model yourself, the Example training script is located in ~/src/main_train.py

To train the model, you should firstly make sure you have downloaded the training datasets Sfm120k or GoogleLandmarksv2 in  ~/data/train/, then you can start the training with the settings described in the paper by running

```ruby
   python3 -m main_train [-h] [--training-dataset DATASET] [--no-val]
                [--test-datasets DATASETS] [--test-whiten DATASET]
                [--test-freq N] [--arch ARCH] [--pool POOL]
                [--local-whitening] [--regional] [--whitening]
                [--not-pretrained] [--loss LOSS] [--loss-margin LM]
                [--image-size N] [--neg-num N] [--query-size N]
                [--pool-size N] [--gpu-id N] [--workers N] [--epochs N]
                [--batch-size N] [--optimizer OPTIMIZER] [--lr LR] [--ld LD]
                [--soa] [--weight-decay W] [--soa-layers N] [--sos] [--lambda N] 
                [--print-freq N] [--flatten-desc]
                EXPORT_DIR
```
</p>
</details>

<details><summary><b>Test</b></summary>
<p>
Firstly, please make sure you have downloaded the test datasets and put them under ~/data/test/.
Then you can start retrieval tests as following:


### Testing on R-Oxford, R-Paris

```ruby
   python3 -m ~src.main_retrieve
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.
### Testing with the extra 1-million distractors
```ruby
   python3 -m ~src.extract_1m
   python3 -m ~src.test_1m
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.
### Testing on Custom
```ruby
   python3 -m ~src.test_custom
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.

### Testing on GoogleLandmarks v2 test
```ruby
   python3 -m ~src.test_GLM
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.

### Testing re-ranking methods
You can use three re-ranking methods (QGE, SAHA, and LoFTR) in any datasets in the following python files:
```ruby
   python3 -m ~src.test_extracted # This is an example of our pipeline. You can test any datasets with this file.
   python3 -m ~src.server   # This is our pipeline with GUI.
```
These two python files can help you to use re-ranking.  

By these files, you can test extracted features from any dataset. You can put preextracted features under this path: src/outputs/. And please unzip the file (utils_files.zip) in "src/utils/" before using.  

And please check paths in "test_extracted.py", "server.py", and "Reranking" (under "src/utils/") before using. You need to set your own paths on a Linux server or your local computer.   

The pretrained feature extraction weight: https://drive.google.com/file/d/1fylhFYW0vYIBpYts_bx4IMiIPL34V5Yb/view?usp=sharing   

You can put rhe weight under this path: src/EXPORT_DIR_QZ/resnet101-gem-w-tri/

To test re-ranking methods, you can use the following api in the aforementioned two files:

For QGE:
```ruby
QGE(ranks, qvecs, vecs, dataset, gnd, query_num, cache_dir, gnd_path2, RW, AQE)  
```
For SAHA: 
```ruby
sift_online(query_num, qimages, sift_q_main_path, images, sift_g_main_path, ranks, dataset, gnd)  
```
For LoFTR: 
```ruby
loftr(loftr_weight_path, query_num, qimages, ranks, images, dataset, gnd)  
```
If you want to use LoFTR, you need to download the pretrained LoFTR weight from: https://github.com/zju3dv/LoFTR  
You can put the LoFTR weight under this path: src/utils/weights/  

You can find detailed annotations about how to use these re-ranking methods in Reranking.py, test_extracted.py and server.py.  

</p>
</details>
<details><summary><b>Retrieval Engine</b></summary>
<p>


</p>
</details>
