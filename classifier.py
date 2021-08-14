import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve

ARRAY_NUMBER = 1


class Classifier:
    """Class used to classify images

    Parameters
    ----------
    image : array_like or list
        The main image of the class
    features : str, optional
        Which features should be used during convolution, by default "all"
        Features include:
            - edges
            - corners
            - curves
            - all (edges, corners, and curves)
    name : str, optional
        The name of the image/main class, by default None

    """
    def __init__(self, image, features="all", name=None):
        global ARRAY_NUMBER
        self.image = image
        self.feature_list = {
            "edges": self.__get_edges(),
            "corners": self.__get_corners(),
            "curve": self.__get_curve(),
            "all": self.__get_edges() + self.__get_curve() + self.__get_corners()
        }
        self.features = self.feature_list[features]

        self.name = self.__generate_name() if name == None else name
        
        self.classes = {}
    
    def __get_edges(self):
        return [
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]]
        ]

    def __get_corners(self):
        return [
            [[1, 1], [1, 0]],
            [[1, 1], [0, 1]],
            [[1, 0], [1, 1]],
            [[0, 1], [1, 1]]
        ]

    def __get_curve(self):
        return [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]]
        ]
    
    def __generate_name(self):
        global ARRAY_NUMBER
        name =  f'array{ARRAY_NUMBER}'
        ARRAY_NUMBER += 1
        return name

    def convolve(self, image=None, show=False, name=None):
        """Convolve over an image with certain features

        Parameters
        ----------
        image : array_like, optional
            A seprate image to convolve, by default None
        show : bool, optional
            Show the final result, by default False
        name : str, optional
            The name that will be returned, by default None

        Returns
        -------
        tuple
            A tuple of the final result and the name of the image
        """
        global ARRAY_NUMBER

        # the convolved list
        convolved = []

        # go through the features
        for feature in self.features:
            if type(image) == np.ndarray:
                # whether or not to convolve the input image
                convolved.append(convolve(image, feature, method='direct'))
            else:
                # convolving the image which was input in the __init__ method
                convolved.append(convolve(self.image, feature, method='direct'))

        # show the final product
        SUM = np.sum(convolved, axis=0)
        if show:
            plt.imshow(SUM)
            plt.show()

        # basically return it with a different name based on circumstances
        if name and type(image) == np.ndarray:
            return (SUM, name)
        elif type(image) == np.ndarray:
            return (SUM, self.__generate_name())
        elif name:
            return (SUM, name)
        else:
            return (SUM, self.name)

    def mass_convolve(self, images, show=False, names=None, inception_convolve=False):
        """Convolves on multiple images

        Parameters
        ----------
        images : list
            A list of all the images
        show : bool, optional
            Whether or not to show the images, by default False
        names : list, optional
            A list of names for the images, by default None
        inception_convolve : int, optional
            Whether to convolve images multiple times (How many times to convolve), by default False

        Returns
        -------
        list
            A list of convolved images
        """
        convolved = []
        for image in images:
            if names:
                for name in names:
                    if inception_convolve:
                        convolved.append(self.inception_convolve(inception_convolve, image, show=show, name=name))
                    else:
                        convolved.append(self.convolve(image, show=show, name=name))
            else:
                if inception_convolve:
                    convolved.append(self.inception_convolve(inception_convolve, image, show=show))
                else:
                    convolved.append(self.convolve(image, show=show))

        return convolved

    def inception_convolve(self, n_times, image, show=False, name=None):
        """Convolve a convolved image n times

        Parameters
        ----------
        n_times : int
            How many times it will be convolved
        image : array_like
            The image to convolve
        show : bool, optional
            Whether or not to show the final product, by default False
        name : str, optional
            The name of the image, by default None
        """
        if n_times == 1:
            return self.convolve(image, name=name)[0]
        else:
            return self.convolve(self.inception_convolve(n_times-1, image, show=show, name=name), show=show, name=name)
        
    def get_classification_percent(self, image, test_image, print_output=True, names=None):
        """Get the percent that two arrays are matching

        Parameters
        ----------
        image : tuple or array_like or list
           The image that is being compared
        test_image : tuple or array_like or list
            The image that will be used to compare
        print_output : bool, optional
            Whether to print the output, by default True
        names : list, optional
            A list of names if the images aren't tuples, by default None

        Returns
        -------
        float
            The percent of the array being the same as the other
        """
        if type(image) == tuple and type(test_image) == tuple:
            total_matches = []
            for lst in image[0]:
                matches = []
                for lst2 in test_image[0]:
                    # getting the matching elements in both lists
                    match = set(lst) & set(lst2)
                    # finding out where all the matches are
                    where = list(itertools.chain.from_iterable([np.where(lst == x)[0] for x in match]))
                    # getting how many matches were found divided by the length of the original list
                    matches.append(len(set(where))/len(lst))
                total_matches.append(max(matches))
            # Average percent out of all of them
            final_prob = sum(total_matches)/len(total_matches)
            if print_output:
                if names:
                    string = f"Percent of {names[0]} being {names[1]}: {round(final_prob*100, 1)}%"
                    print(string)
                    print("_"*len(string))
                else:
                    string = f"Percent of {test_image[1]} being {image[1]}: {round(final_prob*100, 1)}%"
                    print(string)
                    print("_"*len(string))

            return final_prob
        else:
            # for when it's not a tuple
            # ex. (convolved_image, name)
            total_matches = []
            for lst in image:
                matches = []
                for lst2 in test_image:
                    match = set(lst) & set(lst2)
                    where = list(itertools.chain.from_iterable([np.where(lst == x)[0] for x in match]))
                    matches.append(len(set(where))/len(lst))
                total_matches.append(max(matches))
            final_prob = sum(total_matches)/len(total_matches)
            if print_output:
                if names:
                    string = f"Percent of {names[0]} being {names[1]}: {round(final_prob*100, 1)}%"
                    print(string)
                    print("_"*len(string))
                else:
                    string = f"Percent of test_image being main image: {round(final_prob*100, 1)}%"
                    print(string)
                    print("_"*len(string))

            return final_prob

    def classify(self, test_image, classes=None, actual_class=None):
        """Classify an image

        Parameters
        ----------
        test_image : tuple or array_like
            The image that will be comparing
        classes : dict[str, list]
            A dictionary of classes with the corresponding images under each class
        actual_class : str or int, optional
            The actual class of the test image, by default None
        """
        if classes == None:
            classes = self.classes
        else:
            self.classes = classes

        if type(test_image) == tuple:
            class_percents = {}
            for class_name, images in zip(classes.keys(), classes.values()):
                class_percent = []
                for image in images:
                    # get the percent for each image in each class by using the.get_classification_percent method above
                    percent = self.get_classification_percent(image[0], test_image[0], False, [test_image[1], class_name])
                    class_percent.append(percent)
                class_percents.update({class_name:class_percent})

            # get the index of the class that has the higher probability for the image
            most_likely_class = list(class_percents.keys())[list(class_percents.values()).index(max(class_percents.values()))]
            print(f"{test_image[1]} most likely belongs to {most_likely_class}")
            if actual_class:
                print(f"Actual Class: {actual_class}")
                if actual_class == most_likely_class:
                    print(f"Classification is correct!")
                    # store it in the classes variable so it grows and gets better with each item you test
                    self.classes[most_likely_class].append((test_image[0], most_likely_class))
                else:
                    print(f'Classification wrong.')
                    self.classes[most_likely_class].append((test_image[0], actual_class))
                    
            print("_"*50)

        else:
            # for when it's not a tuple
            class_percents = {}
            for class_name, images in zip(classes.keys(), classes.values()):
                class_percent = []
                for image in images:
                    percent = self.get_classification_percent(image, test_image, False, [self.__generate_name(), class_name])
                    class_percent.append(percent)
                class_percents.update({class_name:class_percent})

            most_likely_class = list(class_percents.keys())[list(class_percents.values()).index(max(class_percents.values()))]
            print(f"Test image most likely belongs to {most_likely_class}")
            if actual_class:
                print(f"Actual Class: {actual_class}")
                if actual_class == most_likely_class:
                    print(f"Classification is correct!")
                    self.classes[most_likely_class].append((test_image, most_likely_class))
                else:
                    print(f'Classification wrong.')
                    self.classes[most_likely_class].append((test_image, actual_class))
            print("_"*50)
        return most_likely_class
    
    def find_object_position(self, image, condition="mean"):
        """Find the object's position in an image.

        Parameters
        ----------
        image : array_like or list
            The image that has the object
        condition : lambda function
            A condition that will determine whether or not a pixel is classified as having nothing inside it.
        """
        if condition == "mean":
            MEAN = np.mean([sum(x)/len(x) for x in zip(*image)])
            condition = lambda x: x<MEAN
            indexes = [np.where(condition(row))[0] for row in image]
        else:
            indexes = [np.where(condition(row))[0] for row in image]
        max_index = indexes[[len(r) for r in indexes].index(max(len(r) for r in indexes))]
        object_position = []
        for row in range(len(image)):
            # w = sorted(indexes[row])
            if len(indexes[row]) > 0:
                object_position.append(np.ndarray.tolist(image[row][max_index[0]:max_index[-1]+1]))
                #[w[0]:w[-1]+1]
                # except: ...
        return object_position
    
    def find_and_classify(self, image, original_image, classes, actual_class=None):
        # find more above average or below average
        class_ = self.classify(image, self.classes, actual_class=actual_class)
        plt.imshow(self.find_object_position(original_image, "mean"))
        plt.title(class_)
        plt.show()

    

image = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0]
])

image2 = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1]
])

image3 = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1]
])

image4 = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

image5 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

image6 = np.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0]
])

image6 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0]
])

image8 = np.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0]
])

image9 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

classifier = Classifier(image, "all", "Circle")
new_image1 = classifier.convolve()

new_image2 = classifier.convolve(image2, name="Square")

new_image3 = classifier.convolve(image3, name="test1")

new_image4 = classifier.convolve(image4, name="test2")

new_image5 = classifier.convolve(image5, name="test_square")
new_image6 = classifier.convolve(image6, name="test_circle")
new_image7 = classifier.convolve(IMAGE, name="28x28Circle")
new_image8 = classifier.convolve(image8, name="fat_circle")
new_image9 = classifier.convolve(image9, name="fat_square")


classifier.get_classification_percent(new_image1, new_image3, False)
classifier.get_classification_percent(new_image2, new_image3, False)

classifier.get_classification_percent(new_image1, new_image4, False)
classifier.get_classification_percent(new_image2, new_image4, False)

classifier.classify(new_image3, {"square": [new_image2, new_image9], "circle": [new_image1, new_image8]}, "square")
classifier.classify(new_image4, actual_class="circle")
classifier.classify(new_image5, actual_class="square")
classifier.classify(new_image7, actual_class="circle")
classifier.classify(new_image8, actual_class="circle")
classifier.classify(new_image9, actual_class="square")
