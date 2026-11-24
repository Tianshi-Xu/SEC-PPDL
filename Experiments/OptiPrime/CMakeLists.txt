add_library(QuantExperiment STATIC src/Bundle.cpp src/PreparedOT.cpp src/Context.cpp src/Protocols.cpp src/Model.cpp src/SharedSelection.cpp)
target_include_directories(QuantExperiment PUBLIC include)
target_link_libraries(QuantExperiment PUBLIC LinearLayer NonlinearOperator)
add_executable(test_quant_imagenet src/TestImageNet.cpp)
target_link_libraries(test_quant_imagenet PRIVATE QuantExperiment)

if(NOT BUILD_TESTING)
    return()
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(test_prepared_ot src/TestPreparedOT.cpp)
target_link_libraries(test_prepared_ot PRIVATE QuantExperiment)
add_executable(test_quant_protocols src/TestProtocols.cpp)
target_link_libraries(test_quant_protocols PRIVATE QuantExperiment)
add_executable(test_quant_he src/TestHEParameters.cpp)
target_link_libraries(test_quant_he PRIVATE QuantExperiment)
foreach(backend iknp ferret)
    foreach(prepared 0 1)
        set(test_name quant_ot_${backend}_${prepared})
        add_test(NAME ${test_name}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
                --kind ot --binary $<TARGET_FILE:test_prepared_ot>
                --ot ${backend} --preprocess ${prepared}
                --output ${CMAKE_BINARY_DIR}/test-results/${test_name})
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
    endforeach()
    set(test_name quant_protocols_${backend})
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
            --kind protocols --binary $<TARGET_FILE:test_quant_protocols>
            --ot ${backend} --preprocess 1 --threads 4
            --output ${CMAKE_BINARY_DIR}/test-results/${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
endforeach()
add_test(NAME quant_he_cpu
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
        --kind he --binary $<TARGET_FILE:test_quant_he> --linear cpu
        --output ${CMAKE_BINARY_DIR}/test-results/quant_he_cpu)
set_tests_properties(quant_he_cpu PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
add_test(NAME quant_po2_algebra
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify_po2.py)
foreach(backend iknp ferret)
    add_test(NAME quant_po2_protocols_${backend}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
            --kind protocols --binary $<TARGET_FILE:test_quant_protocols>
            --ot ${backend} --preprocess 1 --po2 1 --threads 7
            --output ${CMAKE_BINARY_DIR}/test-results/quant_po2_protocols_${backend})
    set_tests_properties(quant_po2_protocols_${backend} PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
endforeach()
add_test(NAME quant_po2_he_cpu
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
        --kind he --binary $<TARGET_FILE:test_quant_he> --linear cpu --po2 1
        --output ${CMAKE_BINARY_DIR}/test-results/quant_po2_he_cpu)
set_tests_properties(quant_po2_he_cpu PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
add_test(NAME quant_narrow_prime_protocols
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
        --kind protocols --binary $<TARGET_FILE:test_quant_protocols>
        --ot ferret --preprocess 1 --accumulator-bits 26 --threads 7
        --output ${CMAKE_BINARY_DIR}/test-results/quant_narrow_prime_protocols)
set_tests_properties(quant_narrow_prime_protocols PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
foreach(mode prime po2)
    if(mode STREQUAL "po2")
        set(po2_switch 1)
    else()
        set(po2_switch 0)
    endif()
    add_test(NAME quant_20bit_protocols_${mode}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
            --kind protocols --binary $<TARGET_FILE:test_quant_protocols>
            --ot ferret --preprocess 1 --accumulator-bits 20 --plaintext-bits 20
            --po2 ${po2_switch} --po2-bits 20 --threads 7 --linear cpu
            --output ${CMAKE_BINARY_DIR}/test-results/quant_20bit_protocols_${mode})
    set_tests_properties(quant_20bit_protocols_${mode} PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
endforeach()
add_test(NAME quant_32bit_protocols_po2
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
        --kind protocols --binary $<TARGET_FILE:test_quant_protocols>
        --ot ferret --preprocess 1 --po2 1 --po2-bits 32 --threads 4
        --output ${CMAKE_BINARY_DIR}/test-results/quant_32bit_protocols_po2)
set_tests_properties(quant_32bit_protocols_po2 PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
if(USE_HE_GPU)
    add_test(NAME quant_he_gpu
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
            --kind he --binary $<TARGET_FILE:test_quant_he> --linear gpu
            --output ${CMAKE_BINARY_DIR}/test-results/quant_he_gpu)
    set_tests_properties(quant_he_gpu PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
    add_test(NAME quant_po2_he_gpu
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run.py
            --kind he --binary $<TARGET_FILE:test_quant_he> --linear gpu --po2 1
            --output ${CMAKE_BINARY_DIR}/test-results/quant_po2_he_gpu)
    set_tests_properties(quant_po2_he_gpu PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
endif()
