# 1
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Extracting key and bucket from the event
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from S3 to /tmp/image.png
    download_path = "/tmp/image.png"
    s3.download_file(bucket, key, download_path)
    
    # Read the data from the file and encode it as base64
    with open(download_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')  # Decode to string
    
    # Pass the data back to the Step Function
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []  # Placeholder for additional processing
        }
    }



#2 
import json
import base64
import boto3
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model endpoint
ENDPOINT_NAME = "image-classification-2024-07-14-09-25-15-878"
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    """A function to classify image data using SageMaker"""

    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a Predictor
    predictor = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='image/png',
        Body=image
    )

    # For this model, use IdentitySerializer for "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction
    inferences = predictor['Body'].read().decode('utf-8')

    # Pass the inferences back to the Step Function
    event["inferences"] = inferences

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


#3
import json
THRESHOLD = .93

def lambda_handler(event, context):
    # Get the inferences from the event
    inferences = event["body"]["inferences"]
    
    # Check if any values in any inferences are above THRESHOLD
    meets_threshold = (max(inferences) > THRESHOLD)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }