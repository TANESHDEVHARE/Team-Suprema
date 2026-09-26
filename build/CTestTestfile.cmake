# CMake generated Testfile for 
# Source directory: D:/SIH 2026/trial-2/sovereign_cpp
# Build directory: D:/SIH 2026/trial-2/sovereign_cpp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(run_checks "D:/SIH 2026/trial-2/sovereign_cpp/build/Debug/run_checks.exe")
  set_tests_properties(run_checks PROPERTIES  _BACKTRACE_TRIPLES "D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;89;add_test;D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(run_checks "D:/SIH 2026/trial-2/sovereign_cpp/build/Release/run_checks.exe")
  set_tests_properties(run_checks PROPERTIES  _BACKTRACE_TRIPLES "D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;89;add_test;D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(run_checks "D:/SIH 2026/trial-2/sovereign_cpp/build/MinSizeRel/run_checks.exe")
  set_tests_properties(run_checks PROPERTIES  _BACKTRACE_TRIPLES "D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;89;add_test;D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(run_checks "D:/SIH 2026/trial-2/sovereign_cpp/build/RelWithDebInfo/run_checks.exe")
  set_tests_properties(run_checks PROPERTIES  _BACKTRACE_TRIPLES "D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;89;add_test;D:/SIH 2026/trial-2/sovereign_cpp/CMakeLists.txt;0;")
else()
  add_test(run_checks NOT_AVAILABLE)
endif()
