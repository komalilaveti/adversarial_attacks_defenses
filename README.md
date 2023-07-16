# adversarial_attacks_defenses
**Abstract**<br>
Artificial intelligence (AI) and deep learning (DL) techniques are widely used in various fields such as image classification, object detection, speech recognition, NLP etc. On the other hand, these models especially deep neural networks can easily be fooled by different adversarial attacks. Adversarial attacks involve adding small perturbations to inputs with the goal of getting a machine learning or deep learning model to misclassifying the output. Hence, they bring serious security risks to deep-learning-based systems. So, it is extremely important to provide robustness to deep learning algorithms against these adversaries. In general, adversarial attacks are happened by generating adversarial examples. Adversarial examples are that which may be imperceptible to the human eye, but can lead the model to misclassify the output. In this paper, some of the methods for generating adversarial examples like Fast Gradient Sign Method (FGSM) and Carlini and Wagner (C&W) attacks and few defending methods like adversarial training, autoencoder and a proposed model to be used.
<img width="565" alt="image" src="https://github.com/komalilaveti/adversarial_attacks_defenses/assets/109876090/5543ef18-bcb7-4729-8e95-c6f18b288767">
#Methodology
In this project, image classification model is attacked by some of the adversarial attacks and then defended by the defense techniques. For the image classification model, we consider the dataset GTSRB(German Traffic Sign Recognition Benchmark) which consists of 43 classes of different traffic signs with a total of 39,209 images.
For image classification model, we use L-net model which uses tensorflow.keras sequential model which consists of 7 layers. We train the model with the above dataset and record the accuracy of the model. By considering the trained model, attacking techniques are applied like FGSM (Fast Gradient Sign Method) and C&W (carlini and Wager) attacks which are white box attacks which can access the model. After applying the attacks on the model, the model works with reduced accuracy.
To enhance the reduced accuracy of the model, we apply defense techniques such as adversarial training and autoencoder which try to remove the peturbations added to the images by the attack methods which inturn increases the accuracy of the model.
#Algorithms
1.FGSM(Fast gradient sign method)
2.C&W(carlini and wagner)
3.Adversarial Training
4.Autoencoder
#Results
<img width="231" alt="image" src="https://github.com/komalilaveti/adversarial_attacks_defenses/assets/109876090/41abe10e-4202-4cee-b941-6b9668f85114"> <img width="226" alt="image" src="https://github.com/komalilaveti/adversarial_attacks_defenses/assets/109876090/7d9cf352-324c-4ea1-8eec-d2c3a227d0cf">
<img width="649" alt="image" src="https://github.com/komalilaveti/adversarial_attacks_defenses/assets/109876090/582754d7-4b31-46d2-9e3d-d27a60594897">
#Conclusion
DNN algorithms are employed in many fields but the fact that they are vulnerable to adversarial attacks lead to a challenging problem. So, Adversarial attacks in the real world have drawn a lot of attention lately.In this paper some of the adversarial attack methods such as FGSM and C&W attacks which reduced the accuracy of the model to 2% and reduced the confidence levels. Also, some of the defense strategies are applied to increase the accuracy of the model such as adversarial training and autoencoder which revived the accuracy of the image classification model.



