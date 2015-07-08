VERSION = 1.1
CC = clang++
CPPFLAGS = -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -Wall

DEPS = 	image.h \
		rgb.h \
		rgba.h \
		ycrcb.h \
		hsi.h \
		grayscale.h \
		typeconversion.h

OBJ = 	main.o \
		image.o \
		rgb.o \
		rgba.o \
		ycrcb.o \
		hsi.o \
		grayscale.o \
		typeconversion.o

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
LDFLAGS = -L/opt/local/lib
IDFLAGS = -I/opt/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

ImageProcessing: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

# clean:
# 	rm *.o