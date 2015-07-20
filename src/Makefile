VERSION = 1.0
CC = clang++
CPPFLAGS = -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -Wall

DEPS =	main.h \
		Image/Image.h \
		Image/GrayscaleImage.h \
		Image/RGBImage.h \
		Image/RGBAImage.h \
		Image/HSIImage.h \
		Image/YCrCbImage.h \
		EdgeDetector/EdgeDetector.h \
		LinearFilter/LinearFilter.h \
		MorphologicalFilter/MorphologicalFilter.h \
		TypeConversion/TypeConversion.h \
		Parameter/Parameter.h \
		Util/Util.h \
		HuberROFModel/HuberROFModel.h \
		ImageInpainting/ImageInpainting.h \
		RealTimeMinimizer/RealTimeMinimizer.h \
		TVL1Model/TVL1Model.h \
		PrimalDualAlgorithm/Vectors.h \
		PrimalDualAlgorithm/PrimalDualAlgorithm.h

OBJ =	main.o \
		Image/Image.o \
		Image/GrayscaleImage.o \
		Image/RGBImage.o \
		Image/RGBAImage.o \
		Image/HSIImage.o \
		Image/YCrCbImage.o \
		EdgeDetector/EdgeDetector.o \
		LinearFilter/LinearFilter.o \
		MorphologicalFilter/MorphologicalFilter.o \
		TypeConversion/TypeConversion.o \
		Parameter/Parameter.o \
		Util/Util.o \
		HuberROFModel/HuberROFModel.o \
		ImageInpainting/ImageInpainting.o \
		RealTimeMinimizer/RealTimeMinimizer.o \
		TVL1Model/TVL1Model.o \
		PrimalDualAlgorithm/Vectors.o \
		PrimalDualAlgorithm/PrimalDualAlgorithm.o

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
LDFLAGS = -L/opt/local/lib
IDFLAGS = -I/opt/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

iPaur: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

# clean:
# 	rm *.o