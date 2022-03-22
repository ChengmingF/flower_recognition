import os
import shutil
import random

class FlowerDataset():

    def __init__(self, original_dataset_path):
        self.original_dataset_path = original_dataset_path
        self.dataset_path = os.path.join(os.getcwd(), 'dataset')
    def dataset_is_existed(self):
        if not os.path.exists(self.dataset_path):
            self.train_test_split()
            print('Dataset is created successfully. \n')
        else:
            print('Dataset already exists. If you want new dataset, please delete the existed dataset folder. \n')


    def train_test_split(self):
        self.create_folders()
        label_list = os.listdir(self.original_dataset_path)

        for label in label_list:
            image_path = os.path.join(self.original_dataset_path, label)
            self.image_path = image_path
            image_list = os.listdir(image_path)

            #split train test dataset
            test_percent = 0.2
            train_percent = 0.8

            num_test = int(len(image_list) * test_percent)
            test_list = random.sample(image_list, num_test)
            train_list = []

            for img in image_list:
                if img not in test_list:
                    train_list.append(img)

            self.move_image(train_list, label, 'train')
            self.move_image(test_list, label, 'test')

            #remove one image to make each batch better
        list = os.listdir(train_dandelion_path)
        remove_one = random.sample(list, 1)
        os.remove(os.path.join(train_dandelion_path, remove_one[0]))


    def create_folders(self):
        #create the new dataset path
        dataset_path = os.path.join(os.getcwd(), 'dataset')
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        #create the training dataset path
        global train_image_path
        train_image_path = os.path.join(dataset_path, 'train')
        if not os.path.exists(train_image_path):
            os.makedirs(train_image_path)
        self.train_image_path = train_image_path
        #create the training dataset path for daisy
        train_daisy_path = os.path.join(train_image_path, 'daisy')
        if not os.path.exists(train_daisy_path):
            os.makedirs(train_daisy_path)
        #create the training dataset path for dandelion
        global train_dandelion_path
        train_dandelion_path = os.path.join(train_image_path, 'dandelion')
        if not os.path.exists(train_dandelion_path):
            os.makedirs(train_dandelion_path)
        #create the training dataset path for rose
        train_rose_path = os.path.join(train_image_path, 'rose')
        if not os.path.exists(train_rose_path):
            os.makedirs(train_rose_path)
        #create the training dataset path for sunflower
        train_sunflower_path = os.path.join(train_image_path, 'sunflower')
        if not os.path.exists(train_sunflower_path):
            os.makedirs(train_sunflower_path)
        #create the training dataset path for tulip
        train_tulip_path = os.path.join(train_image_path, 'tulip')
        if not os.path.exists(train_tulip_path):
            os.makedirs(train_tulip_path)



        test_image_path = os.path.join(dataset_path, 'test')
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        self.test_image_path = test_image_path
        #create the training dataset path for daisy
        test_daisy_path = os.path.join(test_image_path, 'daisy')
        if not os.path.exists(test_daisy_path):
            os.makedirs(test_daisy_path)
        #create the training dataset path for dandelion
        test_dandelion_path = os.path.join(test_image_path, 'dandelion')
        if not os.path.exists(test_dandelion_path):
            os.makedirs(test_dandelion_path)
        #create the training dataset path for rose
        test_rose_path = os.path.join(test_image_path, 'rose')
        if not os.path.exists(test_rose_path):
            os.makedirs(test_rose_path)
        #create the training dataset path for sunflower
        test_sunflower_path = os.path.join(test_image_path, 'sunflower')
        if not os.path.exists(test_sunflower_path):
            os.makedirs(test_sunflower_path)
        #create the training dataset path for tulip
        test_tulip_path = os.path.join(test_image_path, 'tulip')
        if not os.path.exists(test_tulip_path):
            os.makedirs(test_tulip_path)

    def move_image(self, list, label, flag):
        old_path = self.image_path
        if flag == 'train':
            new_path = os.path.join(self.train_image_path, label)
        if flag == 'test':
            new_path = os.path.join(self.test_image_path, label)

        for item in list:
            old_item = os.path.join(old_path, item)
            new_item = os.path.join(new_path, item)
            shutil.copy2(old_item, new_item)
