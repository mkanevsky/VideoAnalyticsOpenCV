# Video Analytics OpenCV


## Note: This is a work in progress!


1. clone the .DockerFile to secret.Dockerfile

2. provision form recognizer service on Azure

3. configure your secret keys in secret.Dockerfile

4. build your docker image

```
docker build --rm -f "secret.Dockerfile" -t video-analytics-opencv:latest "."
```

5. run your docker image
```
docker run --rm -it  video-analytics-opencv:latest
```
