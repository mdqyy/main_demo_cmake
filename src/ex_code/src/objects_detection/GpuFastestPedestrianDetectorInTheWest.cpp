#include "GpuFastestPedestrianDetectorInTheWest.hpp"

#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/copy.hpp>


#include "helpers/Log.hpp"



#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>


namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "GpuIntegralChannelsDetector");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "GpuIntegralChannelsDetector");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "GpuIntegralChannelsDetector");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "GpuIntegralChannelsDetector");
}


} // end of anonymous namespace

namespace doppia {

GpuFastestPedestrianDetectorInTheWest::GpuFastestPedestrianDetectorInTheWest(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold, const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   cascade_model_p,
                                   non_maximal_suppression_p, score_threshold, additional_border),
      GpuIntegralChannelsDetector(
          options,
          cascade_model_p, non_maximal_suppression_p,
          score_threshold, additional_border),
      BaseFastestPedestrianDetectorInTheWest(options)
{

    return;
}


GpuFastestPedestrianDetectorInTheWest::~GpuFastestPedestrianDetectorInTheWest()
{
    // nothing to do here
    return;
}


const bool use_fractional_features = false;
//const bool use_fractional_features = true;



void GpuFastestPedestrianDetectorInTheWest::set_image(const boost::gil::rgb8c_view_t &input_view)
{
    const bool input_dimensions_changed =
            ((input_gpu_mat.cols != input_view.width())
             or (input_gpu_mat.rows != input_view.height()));

    cv::Mat output_mat;

    // transfer image into GPU --
    boost::gil::opencv::ipl_image_wrapper input_ipl =
            boost::gil::opencv::create_ipl_image(input_view);

    cv::Mat input_mat(input_ipl.get());
//    input_mat.copyTo(output_mat);
    cv::resize(input_mat,output_mat,cv::Size(IMG_WIDTH,IMG_HEIGHT));

    //        std::cout<<"###########resize#############"<<std::endl;

    //printf("input_ipl.get()->nChannels == %i\n", input_ipl.get()->nChannels);
    //printf("input_mat.type() == %i =?= %i\n", input_mat.type(), CV_8UC3);
    //printf("input_mat.channels() == %i\n", input_mat.channels());
    //printf("input_mat (height, width) == (%i, %i)\n", input_mat.rows, input_mat.cols);

    const bool use_cuda_write_combined = true;
    if(use_cuda_write_combined)
    {
        if((input_rgb8_gpu_mem.rows != output_mat.rows) or (input_rgb8_gpu_mem.cols != output_mat.cols))
        {
            // lazy allocate the cuda memory
            // using WRITE_COMBINED, in theory allows for 40% speed-up in the upload,
            // (but reading this memory from host will be _very slow_)
            // tests on the laptop show no speed improvement (maybe faster on desktop ?)
            input_rgb8_gpu_mem.create(output_mat.size(), output_mat.type(), cv::gpu::CudaMem::ALLOC_WRITE_COMBINED);
            input_rgb8_gpu_mem_mat = input_rgb8_gpu_mem.createMatHeader();
        }

        output_mat.copyTo(input_rgb8_gpu_mem_mat); // copy to write_combined host memory
        input_rgb8_gpu_mat.upload(input_rgb8_gpu_mem_mat);  // fast transfer from CPU to GPU
    }
    else
    {
        input_rgb8_gpu_mat.upload(output_mat);  // from CPU to GPU
    }

    //printf("input_rgb8_gpu_mat.channels() == %i\n", input_rgb8_gpu_mat.channels());

    // most tasks in GPU are optimized for CV_8UC1 and CV_8UC4, so we set the input as such
    cv::gpu::cvtColor(input_rgb8_gpu_mat, input_gpu_mat, CV_RGB2RGBA); // GPU type conversion

    if(input_gpu_mat.type() != CV_8UC4)
    {
        throw std::runtime_error("cv::gpu::cvtColor did not work as expected");
    }






    // set default search range --
    if(input_dimensions_changed or search_ranges.empty())
    {
        compute_search_ranges(boost::gil::rgb8c_view_t::point_t(output_mat.cols, output_mat.rows),
                              scale_one_detection_window_size,
                              search_ranges);

        // update the detection cascades
        compute_scaled_detection_cascades();
        set_gpu_scale_detection_cascades();

        // update additional, input size dependent, data
        compute_extra_data_per_scale(output_mat.cols, output_mat.rows);

        // resize helper array
        resized_input_gpu_matrices.resize(search_ranges.size());

        //        compute_search_ranges(input_view.dimensions(),
        //                              scale_one_detection_window_size,
        //                              search_ranges);

        //        // update the detection cascades
        //        compute_scaled_detection_cascades();
        //        set_gpu_scale_detection_cascades();

        //        // update additional, input size dependent, data
        //        compute_extra_data_per_scale(input_view.width(), input_view.height());

        //        // resize helper array
        //        resized_input_gpu_matrices.resize(search_ranges.size());
    } // end of "set default search range"

    return;
}


void GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_detection_cascades()
{
    if(use_fractional_features)
    { // we state fractional cascade stages, instead of the usual cascade stages
        set_gpu_scale_fractional_detection_cascades();
    }
    else
    {
        GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades();
    }
    return;
}


/// This function is a copy+paste+minor_edits of GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades
void GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_fractional_detection_cascades()
{
    // FIXME copy and paste is bad ? Making this a templated function would be better ?
    // (seems more complication for little benefit. Applying the "three is too much" rule,
    // right now, only two copies, so it is ok)

    if(fractional_detection_cascade_per_scale.empty())
    {
        throw std::runtime_error(
                    "GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_detection_fractional_cascades called, but "
                    "fractional_detection_cascade_per_scale is empty");
    }

    const size_t
            cascades_length = fractional_detection_cascade_per_scale[0].size(),
            num_cascades = fractional_detection_cascade_per_scale.size();

    Cuda::HostMemoryHeap2D<fractional_cascade_stage_t>
            cpu_fractional_detection_cascade_per_scale(cascades_length, num_cascades);

    for(size_t cascade_index=0; cascade_index < num_cascades; cascade_index+=1)
    {
        if(cascades_length != fractional_detection_cascade_per_scale[cascade_index].size())
        {
            throw std::invalid_argument("Current version of GpuFastestPedestrianDetectorInTheWest requires "
                                        "multiscales models with equal number of weak classifiers");
            // FIXME how to fix this ?
            // Using cascade_score_threshold to stop when reached the "last stage" ?
            // Adding dummy stages with zero weight ?
        }

        for(size_t stage_index =0; stage_index < cascades_length; stage_index += 1)
        {
            const size_t index = stage_index + cascade_index*cpu_fractional_detection_cascade_per_scale.stride[0];
            cpu_fractional_detection_cascade_per_scale[index] = \
                    fractional_detection_cascade_per_scale[cascade_index][stage_index];
        } // end of "for each stage in the cascade"
    } // end of "for each cascade"

    if(false and ((num_cascades*cascades_length) > 0))
    {
        printf("GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_fractional_detection_cascades "
               "Cascade 0, stage 0 cascade_threshold == %3.f\n",
               cpu_fractional_detection_cascade_per_scale[0].cascade_threshold);
    }

    gpu_fractional_detection_cascade_per_scale.alloc(cascades_length, num_cascades);
    Cuda::copy(gpu_fractional_detection_cascade_per_scale, cpu_fractional_detection_cascade_per_scale);

    return;
}


void GpuFastestPedestrianDetectorInTheWest::compute()
{
    //    std::cout<<"####################################"<<std::endl;

    detections.clear();
    num_gpu_detections = 0; // no need to clean the buffer

    // some debugging variables
    static bool first_call = true;

    assert(integral_channels_computer_p);
    assert(gpu_detection_cascade_per_scale.getBuffer() != NULL);

    const bool use_v1 = true;
    //const bool use_v1 = false;

    // on Jabbah v0 runs at 2.3 Hz, and v1 at 2.55 Hz
    if(use_v1)
    {
        // for each range search
        for(size_t search_range_index=0; search_range_index < search_ranges.size(); search_range_index +=1)
        {
            compute_detections_at_specific_scale_v1(search_range_index, first_call);
        } // end of "for each search range"

        collect_the_gpu_detections();
    }
    else
    { // we use v0

        const bool save_score_image = false;
        //const bool save_score_image = true;

        // for each range search
        for(size_t search_range_index=0; search_range_index < search_ranges.size(); search_range_index +=1)
        {
            compute_detections_at_specific_scale_v0(search_range_index, save_score_image, first_call);
        } // end of "for each search range"

        if(save_score_image)
        {
            // stop everything
            throw std::runtime_error("Stopped the program so we can debug it. "
                                     "See the scores_at_*.png score images");
        }
    }

    log_info() << "number of raw (before non maximal suppression) detections on this frame == "
               << detections.size() << std::endl;

    // windows size adjustment should be done before non-maximal suppression
    if(this->resize_detection_windows)
    {
        (*model_window_to_object_window_converter_p)(detections);
    }

    compute_non_maximal_suppresion();

    first_call = false;

    return;
}
void GpuFastestPedestrianDetectorInTheWest::compute_detections_at_specific_scale_v1(
        const size_t search_range_index,
        const bool first_call)
{
    //    std::cout<<"__________________________________"<<std::endl;

    if(use_fractional_features == false)
    {
        GpuIntegralChannelsDetector::compute_detections_at_specific_scale_v1(
                    search_range_index, first_call);
        return;
    }

    doppia::objects_detection::gpu_integral_channels_t &integral_channels =
            resize_input_and_compute_integral_channels(search_range_index, first_call);

    const ScaleData &scale_data = extra_data_per_scale[search_range_index];

    // const stride_t &actual_stride = scale_data.stride;
    // on current GPU code the stride is ignored, and all pixels of each scale are considered (~x/y_stride == 1E-10)
    // FIXME either consider the strides (not a great idea, using stixels is better), or print a warning at run time

    // compute the scores --
    {
        // compute the detections, and keep the results on the gpu memory
        doppia::objects_detection::integral_channels_detector(
                    integral_channels,
                    search_range_index,
                    scale_data.scaled_search_range,
                    gpu_fractional_detection_cascade_per_scale,
                    score_threshold, use_the_detector_model_cascade,
                    gpu_detections, num_gpu_detections);
    }

    // ( the detections will be colected after iterating over all the scales )

#if defined(BOOTSTRAPPING_LIB)
    current_image_scale = 1.0f/search_ranges[search_range_index].detection_window_scale;
#endif

    return;
}



} // end of namespace doppia
