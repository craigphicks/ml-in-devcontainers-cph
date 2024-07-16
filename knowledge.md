
# choose the right ec2 instance type

Start with the smallest instance type that meets your needs, and then adjust as necessary.  The almost free type is currently

[https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html](is ecs gpu needed?)

# (own) password protect the aws privded private key file

```
ssh-keygen -p -f /home/craig/.ssh/awskeyfile.pem
```

# then the password is required to use the key file
# so the passprot keyword can be safely stored in the container

# (own|container) ssh to ec2 instance

`ssh` will prompt for the password

```
ssh -i ~/.ssh/awskeyfile.pem ec2-user@ec2-18-117-222-69.us-east-2.compute.amazonaws.com
```

Or use "connect" button in the AWS instances web page - then the pem key is not used.  A command line interface will appear.

Note: Using `sudo reboot` from linux terminal seems to cause a problem when trying to log in again.

This might be a workaround:  https://serverfault.com/a/599141


# (aws) install docker on ec2 instance

-  Go to https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html

    -  Follow instructions under "Installing Docker on AL2023"

# aws mfa for root user

https://docs.aws.amazon.com/IAM/latest/UserGuide/enable-virt-mfa-for-root.html


# aws console home

https://us-east-2.console.aws.amazon.com/console/home?region=us-east-2#


# run a container on ec2 instance

To run a container (like the one mentioned in your log excerpt) on AWS, you can use Amazon ECS (Elastic Container Service) with Fargate, which allows you to run containers without managing servers or clusters. Here's a step-by-step guide:

Step 1: Create an AWS Account
If you haven't already, sign up for an AWS account at https://aws.amazon.com.

Step 2: Create an ECR Repository
Navigate to the Amazon ECR (Elastic Container Registry) service in the AWS Management Console.
Click "Create repository" and follow the prompts to create a new repository for your container image.
Step 3: Push Your Container Image to ECR
Authenticate Docker to your ECR repository using the AWS CLI:
Tag your Docker image for ECR:
Push the image to ECR:
Step 4: Create a Task Definition in ECS
Go to the Amazon ECS console and select "Task Definitions" > "Create new Task Definition".
Select "Fargate" as the launch type and proceed to configure the task and container definitions. For the container image, use the URI of the image you pushed to ECR.
Step 5: Launch the Container with Fargate
Once the task definition is ready, go to the "Clusters" tab in the ECS console and create a new cluster. Choose the "Networking only" cluster template for Fargate.
After creating the cluster, select "Services" > "Create" and follow the wizard to configure your service. Select the task definition you created earlier.
Configure the network settings, choose the VPC and subnets where your container should run, and set the security groups as needed.
Review and create the service. AWS Fargate will start your container in the specified subnets.
Step 6: Monitor Your Container
You can monitor the status of your container and the Fargate service through the ECS console. Logs can be viewed if you configured the log driver in your task definition to use Amazon CloudWatch Logs.
This guide assumes you have a Docker image ready to be deployed. If your image requires specific configurations (like the NVIDIA CUDA base image mentioned in your log), ensure those configurations are set correctly in your Dockerfile and during the task definition phase.


# bundle this container 


# set up this container on ec2 instance

To set up a container on an EC2 instance, you can follow these steps:

1. Connect to your EC2 instance using SSH:
```
ssh -i ~/.ssh/awskeyfile.pem ec2-user@ec2-18-117-222-69.us-east-2.compute.amazonaws.com
```

2. Update the system packages:
```
sudo yum update -y
```

3. Install Docker:
```
sudo amazon-linux-extras install docker -y
```

4. Start the Docker service:
```
sudo service docker start
```

5. Add the current user to the Docker group:
```
sudo usermod -a -G docker ec2-user
```

6. Pull the container image:
```
docker pull <image_name>
```

7. Run the container:
```
docker run -d <image_name>
```

8. Stop and remove the container (if needed):
```
docker stop <container_id>
docker rm <container_id>
```

These steps will help you set up and run a container on your EC2 instance. Make sure to replace `<image_name>` with the actual name of the container image you want to use.

Note: Remember to replace `~/.ssh/awskeyfile.pem` with the path to your AWS private key file.




# (aws) run a container on ec2 instance

GitHub Copilot
ssh -i ~/.ssh/awskeyfile.pem ec2-user@ec2-18-117-222-69.us-east-2.compute.amazonaws.com

sudo yum update -y sudo amazon-linux-extras install docker -y sudo service docker start sudo usermod -a -G docker ec2-user

docker pull <image_name>

docker run -d <image_name>

docker stop <container_id> docker rm <container_id>

To run a container on an EC2 instance, you can use Docker. Here are the steps:

Step 1: Connect to your EC2 instance using SSH:

