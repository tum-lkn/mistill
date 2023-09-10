//
// Created by patrick on 29.09.22.
//
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>


struct measurement_tbl {
  int ncol = 4;
  int max_num_ts = 7 * 5 * 60 * 1000;
  int idx = 0;
  char filename[256] = "/home/sim/Documents/GitHub/lkn/learned-dvr/ccode/tmp.bin";
  int * timings = nullptr;
};


void write_measurements(struct measurement_tbl &tbl) {
  std::ofstream out(tbl.filename, std::ios::app | std::ios::binary);
  if(out) {
    out.write(reinterpret_cast<const char *>(tbl.timings), (int) sizeof(int) * tbl.idx * tbl.ncol);
    out.close();
    tbl.idx = 0;
  } else {
    std::string s(tbl.filename);
    std::cerr << "Could not open output file " << s << std::endl;
  }
}


void take_measurement(struct measurement_tbl &tbl, int idx, int flag, uint64_t dur, int ns) {
  if(tbl.idx < tbl.max_num_ts) {
    tbl.timings[tbl.idx * tbl.ncol + 0] = idx;
    tbl.timings[tbl.idx * tbl.ncol + 1] = flag;
    tbl.timings[tbl.idx * tbl.ncol + 2] = (int) dur;
    tbl.timings[tbl.idx * tbl.ncol + 3] = ns;
    tbl.idx++;
  }
}


int testMeasurementTbl() {
  int retCode = EXIT_SUCCESS;
  struct measurement_tbl tbl;
  tbl.timings = (int*) calloc(12, sizeof(int));
  tbl.max_num_ts = 3;
  take_measurement(tbl, 1, 5, 10, 1);
  if(tbl.idx != 1) {
    std::cerr << "testMeasurement: Wrong index, expected 1 but got " << tbl.idx << std::endl;
    retCode = EXIT_FAILURE;
  }

  if(retCode == EXIT_SUCCESS) {
    take_measurement(tbl, 2, 6, 1, 2);
    if(tbl.idx != 2) {
      std::cerr << "testMeasurement: Wrong index, expected 2 but got " << tbl.idx << std::endl;
      retCode = EXIT_FAILURE;
    }
  }

  if(retCode == EXIT_SUCCESS) {
    take_measurement(tbl, 2, 6, 1, 2);
    take_measurement(tbl, 2, 6, 1, 2);
    take_measurement(tbl, 2, 6, 1, 2);
    take_measurement(tbl, 2, 6, 1, 2);
    if(tbl.idx != 3) {
      std::cerr << "testMeasurement: Wrong index, expected 3 but got " << tbl.idx << std::endl;
      retCode = EXIT_FAILURE;
    }
  }

  if(retCode == EXIT_SUCCESS) {
    write_measurements(tbl);
    if(tbl.idx != 0) {
      std::cerr << "testMeasurement: Wrong index, expected 2 but got " << tbl.idx << std::endl;
      retCode = EXIT_FAILURE;
    }
    take_measurement(tbl, 3, 3, 3, 3);
    take_measurement(tbl, 3, 3, 3, 3);
    write_measurements(tbl);
  }

  free(tbl.timings);
  return retCode;
}


int main(void) {
  if(testMeasurementTbl()) std::cout << "testMeasurement succesful" << std::endl;
  return EXIT_SUCCESS;
}