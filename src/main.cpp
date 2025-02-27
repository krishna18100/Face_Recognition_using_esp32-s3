#include <Arduino.h>
#include "esp_camera.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"
#include "esp32/hal/gpio.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/softmax.h"

// Include the converted model header file
#include "model_06.h"  // This file is generated by the 'xxd' command
#include "camera_pins.h"
// Define model input dimensions
constexpr int kModelInputWidth = 128;
constexpr int kModelInputHeight = 128;
constexpr int kModelInputChannels = 3;

// Camera configuration
camera_config_t config;

// Pointer to store tensor arena
uint8_t* tensor_arena = NULL;  // We'll allocate this dynamically

// RGB565 to RGB888 conversion function
uint32_t rgb565torgb888(uint16_t color)
{
    uint8_t hb, lb;
    uint32_t r, g, b;

    lb = (color >> 8) & 0xFF;
    hb = color & 0xFF;

    r = (lb & 0x1F) << 3;
    g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
    b = (hb & 0xF8);

    return (r << 16) | (g << 8) | b;
}

// Get image from camera frame buffer and convert it to input tensor
int GetImage(camera_fb_t * fb, TfLiteTensor* input) 
{
    assert(fb->format == PIXFORMAT_RGB565);

    // Trimming Image
    int post = 0;
    int startx = (fb->width - kModelInputWidth) / 2;
    int starty = (fb->height - kModelInputHeight) / 2; // Center the crop
    for (int y = 0; y < kModelInputHeight; y++) {
        for (int x = 0; x < kModelInputWidth; x++) {
            int getPos = (starty + y) * fb->width + startx + x;
            uint16_t color = ((uint16_t *)fb->buf)[getPos];
            uint32_t rgb = rgb565torgb888(color);

            float *image_data = input->data.f;

            image_data[post * 3 + 0] = ((rgb >> 16) & 0xFF) / 255.0f;  // Normalize R
            image_data[post * 3 + 1] = ((rgb >> 8) & 0xFF) / 255.0f;   // Normalize G
            image_data[post * 3 + 2] = (rgb & 0xFF) / 255.0f;          // Normalize B
            post++;
        }
    }
    return 0;
}

void setup() {
    Serial.begin(115200);

    // Check available heap memory (internal SRAM + PSRAM if available)
    Serial.print("Free Heap: ");
    Serial.println(ESP.getFreeHeap());

    // Optionally, print out the PSRAM size if available
    if (psramFound()) {
        Serial.print("PSRAM size: ");
        Serial.println(ESP.getPsramSize());
    }

    // Dynamically allocate tensor arena based on available memory
    size_t arena_size = ESP.getFreeHeap();  // Use all available free heap memory
    if (psramFound()) {
        // Allocate in PSRAM if available
        tensor_arena = (uint8_t*)ps_malloc(arena_size);
    } else {
        tensor_arena = (uint8_t*)malloc(arena_size);
    }

    if (tensor_arena == NULL) {
        Serial.println("Memory allocation failed for tensor arena");
        return;
    }
    Serial.print("Tensor arena allocated with size: ");
    Serial.println(arena_size);

    // Camera initialization
 camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QQVGA;; // Updated to match input size
  config.pixel_format = PIXFORMAT_RGB565;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM; // Use PSRAM for frame buffer
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera initialization failed");
        return;
    }

    // Load TFLite model
    const tflite::Model* model = tflite::GetModel(quantized_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version does not match TFLite Micro runtime");
        return;
    }

    // Set up the interpreter
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, arena_size);
    interpreter.AllocateTensors();

    // Get input and output details
    TfLiteTensor* input_tensor = interpreter.input(0);
    TfLiteTensor* output_tensor = interpreter.output(0);

    if (input_tensor->type != kTfLiteUInt8 && input_tensor->type != kTfLiteFloat32) {
        Serial.println("Unsupported input tensor type");
        return;
    }
}

void loop() {
    // Capture an image from the camera
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }

    // Preprocess and resize the image (128x128)
    GetImage(fb, interpreter.input(0));

    esp_camera_fb_return(fb);

    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        Serial.println("Error during inference");
        return;
    }

    // Get output and print max value
    TfLiteTensor* output_tensor = interpreter.output(0);
    if (output_tensor->type == kTfLiteUInt8) {
        uint8_t* output_data = output_tensor->data.uint8;
        uint8_t max_value = output_data[0];
        for (int i = 1; i < output_tensor->bytes; i++) {
            if (output_data[i] > max_value) {
                max_value = output_data[i];
            }
        }
        Serial.print("Max value in output: ");
        Serial.println(max_value);
    }

    delay(1000);  // Add a delay between inferences
}
