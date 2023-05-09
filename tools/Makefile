#SM=35
#ARCH = -arch sm_$(SM)
all: KE ConvertHelioToBarry Convert_P0T1_to_P1T0 Convert_P1T0_to_P0T1

KE: KE.cpp
	g++ -o KE KE.cpp

ConvertHelioToBarry: ConvertHelioToBarry.cpp
	g++ -o ConvertHelioToBarry ConvertHelioToBarry.cpp

Convert_P0T1_to_P1T0: Convert_P0T1_to_P1T0.cpp
	g++ -o Convert_P0T1_to_P1T0 Convert_P0T1_to_P1T0.cpp

Convert_P1T0_to_P0T1: Convert_P1T0_to_P0T1.cpp
	g++ -o Convert_P1T0_to_P0T1 Convert_P1T0_to_P0T1.cpp

