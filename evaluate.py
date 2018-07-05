import numpy as np
from argparse import ArgumentParser
import pandas as pd


def evaluate_localization(predicted, target):
    """Evaluates localization results.
        Input:
            predicted: matrix of N x 4 real numbers (x,y,u,v)
                       representing the predicted 2D location (x,y)
                       and the orientation (u,v)

            target:    ground truth locations/orientations in the same form of predicted
            
        Output:
            (position_error, orientation_error)"""

    position_errors = np.sqrt(((predicted[:,:2]-target[:,:2])**2).sum(1))

    def normalize(x):
        return x/np.sqrt((x**2).sum(1)).reshape(-1,1)

    predicted_orientations = normalize(predicted[:,2:])
    target_orientations = normalize(target[:,2:])

    orientation_errors = np.degrees(np.arccos((predicted_orientations*target_orientations).sum(1)))

    return position_errors.mean(), np.median(position_errors), np.mean(orientation_errors), np.median(orientation_errors)



def main(predicted_locations, target_locations):
    #parser = ArgumentParser()

    #parser.add_argument('predicted_locations', type=str, help='Path to the csv containing the predicted locations')
    #parser.add_argument('target_locations', type=str, help='Path to the csv containing the ground truth locations')

    #args=parser.parse_args()

    names=['img','x','y','u','v']

    predicted=pd.read_csv(predicted_locations, header=None, names=names)
    target=pd.read_csv(target_locations, header=None, names=names)

    assert len(predicted)==len(target), "The number of predicted and target locations should match"

    predicted.sort_values('img',inplace=True)
    target.sort_values('img',inplace=True)

    predicted = predicted.drop('img',1).as_matrix()
    target = target.drop('img',1).as_matrix()

    errors = evaluate_localization(predicted,target)

    #print "Errors:"
    #print "Mean Location Error: %0.4f" % (errors[0],)
    #print "Median Location Error: %0.4f" % (errors[1],)
    #print "Mean Orientation Error: %0.4f" % (errors[2],)
    #print "Median Orientation Error: %0.4f" % (errors[3],)

    # save on txt file
    with open("errors.txt", "w") as text_file:
        text_file.write("Mean Location Error: {:.4f}\nMedian Location Error: {:.4f}\nMean Orientation Error: {:.4f}\nMedian Orientation Error: {:.4f}"
                    .format(errors[0], errors[1], errors[2], errors[3]))

#if __name__== "__main__":
    #main()
