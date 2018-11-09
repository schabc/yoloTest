import cv2
from os.path import isfile, join
from os import listdir


'''--------Here fnish the minibatch operation--------'''
def images( path ):
    filenames = [join( path, f ) for f in listdir( path ) if isfile( join( path, f ) )]

#    batch_filenames = []
#    num = len( filenames ) // batch_size
#    for i in range( num ):
#        batch_filename = filenames[i * batch_size : ( i + 1 ) * batch_size]
#
#        batch_filenames.append( batch_filename )

    '''--------Ignore some data--------'''
    # if len( filenames ) % batch_size:
    #     batch_filename = filenames[num * batch_size :]

    #     batch_filenames.append( batch_filename )

    return filenames#batch_filenames

def get_image( path, width, height ):
    image = cv2.imread( path )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    # image = np.true_divide( image, 255 )
    image = cv2.resize( image, ( width, height ) )

    return image

def labels( path ):
    # batch_labels = []
    labels_filenames = images( path )
    # for label_filename in labels_filenames:
    #     batch_label = extract_labels.labels_normaliszer( label_filename )
    #     batch_labels.append( batch_label )

    return labels_filenames




'''--------Test images--------'''
if __name__ == '__main__':
    image = images( 3, './THZDataset/VOC2007/JPEGImages/' )

    print( len( image ) )