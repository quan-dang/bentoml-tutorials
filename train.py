from bento_service import IrisClassifier
from sklearn import svm
from sklearn import datasets

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack(
    'model', 
    clf,
    metadata={
        'created_by': 'quan.dang'
    }
)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save(version="0.0.3")