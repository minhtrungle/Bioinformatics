# Bioinformatics
##1. DeepMal: Accurate prediction of protein malonylation sites by deep neural networks

##Pipeline

###DeepMal uses the following dependencies:

**MATLAB2014a

**python 3.6

**numpy

**scipy

**scikit-learn (Deep learning library)

**keras(Machine learning library)
###Guiding principles:

**The data contains training dataset and testing dataset. Training dataset includes ecoli_train,H_train and mus_train Testing dataset includes ecoli_test,H_test and mus_test

**Feature extraction: EAAC.py is the implementation of enhanced amino acid composition. EGAAC.py is the implementation of enhanced grouped amino acid composition. KNN.py is the implementation of K nearest neighbors. DDE.py is the implementation of dipeptide deviation from expected mean. BLOSUM62.py is the implementation of BLOSUM62 matrix.

** Classifier: DL.py is the implementation of DL. DL_1.py is the implementation of DL_1. DNN.py is the implementation of Deep neural network. GRU.py is the implementation of Recurrent neural network. XGBoost_classifier.py is the implementation of XGBoost. SVM_classifier.py is the implementation of SVM.

##2.predict_infection_risk_of_coronavirus

**Using the spike protein feature to predict infection risk and monitor the evolutionary dynamic of coronavirus
1.	Môi trường thử nghiệm

     Chương trình được thử nghiệm trên môi trường Google Colab. Đây là một môi trường notebook Jupyter được host, sử dụng miễn phí và không cần thiết lập trên Google. Với cấu hình như sau:
     
•	CPU: Intel(R) Core(TM) CPU @ 2.30GHz 

•	RAM: 8 GB

     Chương trình sử dụng các thư viện sau:
     
•	Matplotlib

•	Numpy

•	Pandas

•	Sklearn

•	Transformers

•	Tensorflow

•	Và một số thư viện: re, sys, os, …

2.	Cấu trúc thư mục và tên file

a.	Thư mục “predict_infection_risk_of_coronavirus”: sử dụng trích xuất đặc trưng AAC, PseAAC và GGAP

-	Thư mục “FeatureExtraction” chứa chương trình để biểu diễn đặc trưng tương ứng với tên thư mục AAC, PseAAC và GGAP. Mỗi thư mục đều có:
	
•	“tên file trùng tên thư mục. py”: chương trình python để trích xuất đặc trưng

•	“tên file trùng tên thư mục_data.csv”: chứa dữ liệu về đặc trưng sau khi trích xuất.

•	“tên file trùng tên thư mục_data_name.csv”: chứa dữ liệu về đặc trưng sau khi trích xuất kèm thêm cột chứa tên protein.

•	“check_fasta.py”: chương trình kiểm tra file đúng định dạng fasta hay không.

•	“readFasta.py” chương trình python sử dụng để đọc chuỗi file định dạng fasta trả về mảng chứa list có 2 thông tin là tên protein và dữ liệu chuỗi protein.

•	“data.fasta” chứa dữ liệu về chuỗi protein

•	“dev.fasta” chứa dữ liệu về chuỗi protein nhưng với số lượng ít hơn để thuận tiện cho quá trình xây dựng chương trình (giảm thời gian chạy chương trình).

-	Thư mục “Classifier” chứa một số file chính:
	
•	“random_forest_0.9873817034700315.joblib”: chứa dữ liệu về mô hình rừng ngẫu nhiên.

•	“readFasta.py” chương trình python sử dụng để đọc chuỗi file định dạng fasta trả về mảng chứa list có 2 thông tin là tên protein và dữ liệu chuỗi protein.

•	“trainning.py”: chương trình python dùng để huấn luyện và kiểm thử.

•	“use_model.py”: Chương trình python sử dụng file dữ liệu mô hình, đọc file fasta và đưa ra dự đoán.

•	“summary_2.txt”: file chứa log kết quả chương trình huấn luyện

•	“Running....ipynb” chương trình gọi các chương trình python trong các thư mục con để huấn luyện trên colab


