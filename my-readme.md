To enable plotting with matplotlib, run the runcontainer with the following command:
```bash
docker run -ti --network=host -e DISPLAY -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"  -v $(pwd):/workspace/project mnist1
```

