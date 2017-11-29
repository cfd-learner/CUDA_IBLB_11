objects = main.o LatticeBoltzmann.o ImmersedBoundary.o Seconds.o

all: $(objects)
		nvcc -arch=sm_20 $(objects) -o IBLB

%.o: %.cpp
		nvcc -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
		rm -f *.o IBLB