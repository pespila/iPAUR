template<typename aType>
HSIImage<aType>::~HSIImage() {
    free(this->image);
    this->image = NULL;
    this->type = 0;
    this->height = 0;
    this->width = 0;
    this->channels = 0;
}