# CL to improve SASRec

# about the effects:

(1) sas-cts can enhence performance on 2 amazon-datasets;

(2) token-shuffle can further improve the results on 2 amazon-datasets;

(3) DCLR's "noise negative samples" can further improve a very little on 1 amazon-dataset, & no effects on the other amazon-dataset;

(4)ther are no improve when using 2nd-to-last layer of transformer outputs for contrastive learning (2 times of forward to create postive & negative samples);

(5)project-head no gains;

(6)dropout is kind of tricky; the higher dropout, the worse the contrastive-learning-effects; and original SASRec seems usually get highest performance at dropout 0.5 but worse performance on dropout 0.1;

  if compare SASRec & SAS-CTS at dropout 0.5, there is a little gain or no gains; but on dropout 0.1, the SAS-CTS's effects become obvious & seems bigger gain & improvements;
  
  

# install

pip install pyyaml colorlog colorama

# prepare amazon-dataset

in dir: your-path/nnll/exp1/dataset/Amazon_Home_and_Kitchen/

extract the .zip file to obtain .inter file;

# run training config

please check your-path/nnll/exp1/sas_cts.yaml if running run_sas__dtset.py;

for Amazon_Home_and_Kitchen, I have used following batch-size;

"#train_batch_size: 2400 #home-kitchen; 3090-gpu;24G;"

besides, all dropout configs are set to 0.1, for easy compare the original SASRec & SAS-CTS;

# run amazon-dataset

cd your-path/nnll/exp1/

python run_sas__dtset.py

# about the prog config

please check: your-path/nnll/exp1/rstat.py

# about the model file

(1) sascts.py;  changed code mainly in about line283: calculate_loss();

(2)reference prog DCLR;

      your-path/nnll/DCLR-main/simcse/models.py
	  
	  line 596: cl_vat_forward();  impled "noise negative samples";
