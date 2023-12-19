#  MUSIC GENRE CLASSIFICATION SERVICE

![image](https://github.com/dr-zaib/music-genre-classification/blob/main/music_image.jpeg)

## Overview 

This [project](https://github.com/dr-zaib/music-genre-classification) stands as an evaluation test for the [capstone-1 project](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2023/projects.md#capstone-1) from the [Machine Learning Zoomcamp course](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master) delivered by [DataTalks Club](https://datatalks.club/slack.html) conducted by [Alexey Grigorev](https://github.com/alexeygrigorev).


## Problem description

As a singer/song-writer ([find me here](https://open.spotify.com/intl-it/artist/3TxRnp2KxvCWqRu7M9eAc3?si=3r8bTa7-TAS5FaljP_RG-A)), questioning myself on what is the genre of the music I write and sing can be sometimes the most existential dilemma for me in that particular moment...all jokes aside...getting to know what is the genre of my music happens to be really important during the promotional stage.

Yet, it is useful either whenever I need to interact with the Press Office or whenever I have to link up with online playlists curators and/or editors: in this phase, indeed, this *music genre* information makes me save a considerable time, so that I can organize my promotional strategy and take action more efficiently.

So this [project](https://github.com/dr-zaib/music-genre-classification) was born with the idea of offering a service that can classify music, discriminating amongst the available musical genres, along side with the probabilities' predictions on how much close is the music taken into consideration to the different genres.

Among the available genres are: 

- classical
- rock
- metal 
- country 
- jazz
- blues 
- reggae 
- disco 
- pop 
- hiphop

The model can surely be enlarged and trained on more genres, depending on the dataset. 
More information on the dataset used for this scope can be found [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

Therefore, this service can represent an optimal baseline for a bigger standalone service or an application, useful for musicians, not only, but it can also it can be an integrated service to online music promotion platforms or services.




## Project structure

This project constains the following files: 

* [README.md](https://github.com/dr-zaib/music-genre-classification/blob/main/README.md): contains all the info about the project and how to access the service 

[NB: the following script is too big to be here on github, you can visualize it on colab; make sure to select colab as the linked app; you can also see in this [drive folder](https://drive.google.com/drive/folders/1g40ZKjE45oG3U4B1MgyJ5yxVrsgQTo_6?usp=sharing)]
* [notebook.ipynb](https://drive.google.com/file/d/1M4oQkA1Wfd6JHzMihtS9pJj0_jpAuN0M/view?usp=sharing): script with 
    - data preparation 
    - EDA: this analysis has been made genarally and also for each train model
    - model selection process and parameter tuning for each model
    - feature importance analysis for each selected and tuned model

* [extra] [notebook_final.ipynb](https://github.com/dr-zaib/music-genre-classification/blob/main/notebook_final.ipynb): contains the final model being tested on data, the conversion of the model from a TF model to a TFLite one and the corresponding saving process.

* [train.py](https://github.com/dr-zaib/music-genre-classification/blob/main/train.py):
    - training of the final model 
    - saving the model as 'mugen_model_v1_{epoch}_{val_accuracy}.h5' in this case as [mugen_model_v1_175_0.775.h5](https://github.com/dr-zaib/music-genre-classification/blob/main/mugen_model_v1_175_0.775.h5)
    - saving the dataset scaler as [scaler.joblib]()

* [genres_lambda.py](https://github.com/dr-zaib/music-genre-classification/blob/main/genres_lambda.py): 
    - function to be deployed when making a service out of the project; this is the complete TF version
    - prediction function
    - lambda_handler function 

* [genres_lambda_lite.py](https://github.com/dr-zaib/music-genre-classification/blob/main/genres_lambda_lite.py): 
    - TFLite version of the lambda function 
    - prediction function
    - lambda_handler function 

* [model-converter.py](https://github.com/dr-zaib/music-genre-classification/blob/main/model-converter.py): 
    - script to convert the TF model into a TFLite model and saves it as [mugen-model.tflite](https://github.com/dr-zaib/music-genre-classification/blob/main/mugen-model.tflite)

* [requirements.txt](https://github.com/dr-zaib/midterm-project/blob/main/requirements.txt): files with the dependencies for the local virtual environment

* [Dockerfile](https://github.com/dr-zaib/music-genre-classification/blob/main/Dockerfile): file containing the contairnerization info before deploying the service with GCP

* [testing.py](https://github.com/dr-zaib/music-genre-classification/blob/main/testing.py): useful script to test the service


## Access the service 

To access the service just run the file [testing.py](https://github.com/dr-zaib/music-genre-classification/blob/main/testing.py) 
    - from your editor
    - from your terminal
In both cases make sure the environment you are running testing.py the python library [requests](https://pypi.org/project/requests/)


## Data 

* About Dataset 

*Context*

Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

This data hopefully can give the opportunity to do just that.



*Content*

* genres original - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
* images original - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
* 2 CSV files - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.



*Acknowledgements*

* The GTZAN dataset is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions (http://marsyas.info/downloads/datasets.html).
* This was a team project for uni, so the effort in creating the images and features wasn't only my own. So, I want to thank James Wiltshire, Lauren O'Hare and Minyu Lei for being the best teammates ever and for having so much fun and learning so much during the 3 days we worked on this.



* Download the dataset [features_30_sec.csv](https://github.com/dr-zaib/music-genre-classification/blob/main/features_30_sec.csv) from this repository or from [kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)




## Run the project locally

### clone this repository: 

    git clone https://github.com/dr-zaib/music-genre-classification 

### create a anaconda environment: 

        conda create -n name_of_your_env

and activate the environment : 

        conda activate name_of_your_env

you can deactivate it at the end of everything : 

        conda deactivate

### install the dependencies : 

locate into the clone repo : 

        cd music-genre-classification

install the libraries : 

        pip install -r requirements.txt

### train a new model: 

from your terminal:

        python train.py 

this will actually save in your directory different models in the format 'mugen_model_v1_{epoch:02d}_{val_accuracy:.3f}.h5' : choose your prefered model 

also the scaler used to preprocess/scale the data will be saved as 'scaler.jolib'

 you can also run the script train.py in your editor and you will obtain the same result


### predictions: 

* you can have a look on the prediction functioning from the [notebook_final.ipynb](https://github.com/dr-zaib/music-genre-classification/blob/main/notebook_final.ipynb) 

* run the script predict.py in the same ways like as the training script:

    make sure the name of the model is the good one, that is the one you have chosen.

    in case you would need to change data sample for the testing, refer to the [notebook_final.ipynb](https://github.com/dr-zaib/music-genre-classification/blob/main/notebook_final.ipynb)


* lambda function preparation: 

    this function will be deployed on aws; there are 2 versions of it: 
    - the complete TF version: [genres_lambda.py](https://github.com/dr-zaib/music-genre-classification/blob/main/genres_lambda.py) and it contains the model [mugen_model_v1_175_0.775.h5](https://github.com/dr-zaib/music-genre-classification/blob/main/mugen_model_v1_175_0.775.h5)

    - the TFLite version: [genres_lambda_lite.py](https://github.com/dr-zaib/music-genre-classification/blob/main/genres_lambda_lite.py) and it contains the model [mugen-model.tflite](https://github.com/dr-zaib/music-genre-classification/blob/main/mugen-model.tflite)

    to convert the model from 'mugen_model_v1_{epoch}_{val_accuracy}.h5' into a TFLite version of it you can do from your terminal: 

        python model-converter.py

    before you this operation make sure you have loaded the 'mugen_model_v1_{epoch}_{val_accuracy}.h5' into the script

    you can also run the same script in your python editor

    after ensuring that the selected lambda function contains the right model (make sure to import the models you have trained) you can test it locally from your terminal: 

        ipython

    then: 

        import genres_lambda_lite.py 

    then: 

        genres_lambda_lite.predict(X)

    where: 

        X = np.array([[[ 0.8030606 , -0.1557011 ,  1.1055912 ,  0.93655306,
         -0.15369472, -0.11116464, -0.01547439, -0.27911508,
         -0.00249689, -0.10263481, -0.22186784,  0.13539316,
          0.19517972,  0.6543159 , -0.71008056,  1.0272052 ,
          0.60200524,  0.7041032 , -0.41004395,  0.16232735,
         -0.08335441,  0.00821547,  0.26885274,  0.8373929 ,
          0.532862  , -0.684468  , -0.0836993 ,  0.5947242 ,
          0.18089452, -1.1380627 ,  0.5402677 ,  0.8477326 ,
          0.38778695, -0.7326424 , -0.17957942,  1.3638803 ,
         -0.40411943, -1.306172  , -0.30395243,  0.37106547,
         -0.23417114, -1.7178786 , -0.5221558 ,  0.61539984,
         -0.05484645, -1.4934765 , -0.29566684,  0.7736167 ,
          0.4533471 ,  0.22743312,  1.1430705 ,  1.6219116 ,
          1.2821568 , -0.9514257 ,  0.9961895 , -0.6720621 ,
          0.4822668 ]]], dtype='float32')

    you can just copy it!

    you should also test: 

        genres_lambda_lite.lambda_handler(event=data_json, context="context")

    where: 

        data_json = {"data": [[[0.8030605912208557, -0.15570110082626343, 1.1055911779403687, 0.9365530610084534, -0.15369471907615662, -0.1111646369099617, -0.015474390238523483, -0.27911508083343506, -0.002496890025213361, -0.10263480991125107, -0.22186784446239471, 0.1353931576013565, 0.19517971575260162, 0.6543158888816833, -0.7100805640220642, 1.027205228805542, 0.6020052433013916, 0.7041031718254089, -0.41004395484924316, 0.1623273491859436, -0.0833544135093689, 0.008215470239520073, 0.26885274052619934, 0.8373929262161255, 0.5328620076179504, -0.6844679713249207, -0.0836993008852005, 0.594724178314209, 0.18089452385902405, -1.1380627155303955, 0.5402677059173584, 0.8477326035499573, 0.38778695464134216, -0.732642412185669, -0.17957942187786102, 1.3638802766799927, -0.40411943197250366, -1.3061720132827759, -0.3039524257183075, 0.3710654675960541, -0.23417113721370697, -1.7178785800933838, -0.5221558213233948, 0.6153998374938965, -0.05484645068645477, -1.4934765100479126, -0.2956668436527252, 0.7736166715621948, 0.4533470869064331, 0.22743311524391174, 1.1430704593658447, 1.621911644935608, 1.2821568250656128, -0.9514256715774536, 0.9961894750595093, -0.67206209897995, 0.4822668135166168]]]}

    just copy it; you can also find it in the [notebook_final.ipynb](https://github.com/dr-zaib/music-genre-classification/blob/main/notebook_final.ipynb) at the very end!

### containerization and deployment on aws-ecr: 
after all the tests you are ready to containerize everything and then deploy

* in the [Dockerfile](https://github.com/dr-zaib/music-genre-classification/blob/main/Dockerfile) you will find an example of the structure: yet you will have to ensure you RUN every library/dependency is needed for your lambda function, based on if it is full TF or TFLite

    also make sure you have the rigth python base image; you can find it [here at public.ecr.aws](https://gallery.ecr.aws/lambda/python); you just copy and paste it in the Dockerfile

    then from your terminal: 

        docker build -t name_of_your_docker_container . 

    then test it to whether it works: 

        docker run -it --rm -p 8080:8080 name_of_your_docker_container:latest

    you can also check with: 

        docker images

    and you will see all the images available you have created. 


* create an ECR: 

        aws ecr create-repository --repository-name name_of_your_repository 

    you will get a response like this: 

        {
        "repository": {
            "repositoryArn": "arn:aws:ecr:eu-central-1:99999999999:repository/name_of_your_repository",
            "registryId": "99999999999",
            "repositoryName": "name_of_your_repository",
            "repositoryUri": "999999999999.dkr.ecr.eu-central-1.amazonaws.com/name_of_your_repository",
            "createdAt": "2023-12-18T19:13:57+01:00",
            "imageTagMutability": "MUTABLE",
            "imageScanningConfiguration": {
                "scanOnPush": false
            },
            "encryptionConfiguration": {
                "encryptionType": "ZZZ999"
                }
            }
        }

    you are interestested just into the "repositoryUri" part; just copy and the following lines into your terminal, MAKING SURE you have filled them correctly with required info: 

        ACCOUNT=your_account_number 
        REGION=your_region
        REGISTRY=name_of_your_repository
        PREFIX=${ACOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
        TAG=name_of_your_docker_container
        REMOTE_URI=your_account_number.dkr.ecr.eu-central-1.amazonaws.com/name_of_your_repository:name_of_your_docker_image

    then tag your docker image: 

        docker tag your_docker_image:latest ${REMOTE_URI}

    then push it: 

        docker push ${REMOTE_URI}


    at this point you should have successfully created an ECR and pushed it on aws!

* on aws console: 

    - search for ***"lambda"*** in the research bar
    - create a function: select the "container image" option
    - browse the URI image (the ECR you created before)
    - give your function a name and create it
    - test it: paste the value of ***data_json*** into the query box
    - in the configurations box you can adjust the memory size and the timeout


    now you are ready to expose your function: 

    - search for ***"API Gateway"*** in the research bar
    - create new API: 
        - REST API
        - put the name and create
    - create a resource for the API: click on 'create resource', put the name and create
    - for that resource, create a method: 
        - 'create method'
        - type: POST
        - type of integration function: lambda function
    - test it: still, paste the value of ***data_json*** into the query box
    - deploy the API: 
        - new stage
        - put the name
        - deploy

now you should be left with a URL: copy and paste it in the [testing.py](https://github.com/dr-zaib/music-genre-classification/blob/main/testing.py)
then run it to send a query to the service: 

    python testing.py

you can also run it in your editor.

the service is ready to be used!



### proof of deployment 


In case you are not able to see the videos here, check this drive [folder](https://drive.google.com/drive/folders/1g40ZKjE45oG3U4B1MgyJ5yxVrsgQTo_6?usp=sharing) to see them!



![video](https://github.com/dr-zaib/music-genre-classification/blob/main/video1.webm)



![video](https://github.com/dr-zaib/music-genre-classification/blob/main/video2.webm)


## Considerations

Different models had been trained: a MLP, a CNN and 2 different RNN models; amongst these, the CNN have been the lowest performing and it has not been taken into consideration for this project
the **Bidirectional RNN** model instead had revealed itself to be the best. 

![image](https://github.com/dr-zaib/music-genre-classification/blob/main/image_mlp.png)



![image](https://github.com/dr-zaib/music-genre-classification/blob/main/image_rnn1.png)


![image](https://github.com/dr-zaib/music-genre-classification/blob/main/image_rnn2.png)

