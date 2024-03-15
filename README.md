# T-FOLEY: A Controllable Waveform-Domain Diffusion Model for Temporal-Event-Guided Foley Sound Synthesis
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.09294) [![githubio](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://yoonjinxd.github.io/Event-guided_FSS_Demo.github.io/) *Yoonjin Chung, Junwon Lee, Juhan Nam*

This repository contains the implementation of the paper, *[T-FOLEY: A Controllable Waveform-Domain Diffusion Model for Temporal-Event-Guided Foley Sound Synthesis](https://arxiv.org/pdf/2401.09294.pdf)*, accepted in 2024 ICASSP. 

In our paper, we propose ***T-Foley***, a ***T***emporal-event guided waveform generation model for ***Foley*** sound synthesis, which can generate high-quality audio considering both sound class and when sound should be arranged. 


[!img](Architecture_image)

## Inference

To perform inference using our model, follow these steps:

1. Install the required dependencies by running the following command:
    ```
    $ pip install -r requirements.txt
    ```

2. Download the pre-trained model weights and configurations from the following link: [prertrained.zip](TBD).

3. Place the downloaded model weights and config file in the `./pretrained` directory.

4. Run the inference script by executing the following command:
    ```
    $ python inference.py --class_name "DogBark"
    ```

    The class_name **must be** one of the class name of [2023 DCASE Task7 dataset](https://dcase.community/challenge2023/task-foley-sound-synthesis). The list of the class name: `"DogBark", "Footstep", "GunShot", "Keyboard", "MovingMotorVehicle", "Rain", "Sneeze_Cough`

5. The generated samples would be saved in the `./results` directory.


## Training

To train the T-Foley model, follow these steps:

1. Install the required dependencies by running the following command:
    ```
    $ pip install -r requirements.txt
    ```

2. Download and unzip the DCASE_2023_Challenge_Task_7_Dataset:
    ```
    $ wget http://zenodo.org/records/8091972/files/DCASE_2023_Challenge_Task_7_Dataset.tar.gz
    $ tar -zxvf DCASE_2023_Challenge_Task_7_Dataset.tar.gz
    ```
    
    If you use other dataset, prepare file path list of your training data as .csv format and configure to `params.py`.

3. Run the training:
    ```
    $ python train.py
    ```

    This will start the training process and save the trained model weights in the `logs/` directory.

    To see the training on tensorboard, run:
    ```
    $ tensorboard --logdir logs/
    ```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.