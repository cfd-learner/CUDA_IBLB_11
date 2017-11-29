#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "LatticeBoltzmann.cuh"
#include "ImmersedBoundary.cuh"

#include "seconds.h"



using namespace std;

#define XDIM 300
#define YDIM 200

#define Re 1
#define C_S 0.577
#define ITERATIONS 100000
#define INTERVAL 1000
#define RHO_0 1.

#define LENGTH 100
#define T 100000
#define SPEED 0.008

#define PI 3.14159


const double centre[2] = { XDIM/2., 0.};

//const int p_step = nearbyint(T/25);

//const double c_space = LENGTH/2.;

const int c_num = 6;		
const int c_sets = 6/c_num;

const double TAU = (SPEED*LENGTH) / (Re*C_S*C_S) + 1. / 2.;

const double TAU2 = 1. / (12.*(TAU - (1. / 2.))) + (1. / 2.);

//-------------------------------------------PARAMETER SCALING----------------------------

double dt = 1. / (T);
double dx = 1. / LENGTH;

double l_0 = 0.000006;					//6 MICRON CILIUM LENGTH
double t_0 = 0.067;						//67ms BEAT PERIOD AT 15Hz

void print(const double * r, const double * z, const string& directory, const int& time)
{
	unsigned int j(0);

	int x(0), y(0);

	double ab(0);

	string output = directory;
	output += "/";
	output += to_string(time);

	output += "-vector";
	output += ".dat";

	ofstream rawfile(output.c_str());

	rawfile.open(output.c_str(), ofstream::trunc);
	rawfile.close();

	rawfile.open(output.c_str(), ofstream::app);

	
	for (j = 0; j < XDIM*YDIM; j++)
	{
		x = j%XDIM;
		y = (j - j%XDIM) / XDIM;

		ab = sqrt(z[2 * j + 0] * z[2 * j + 0] + z[2 * j + 1] * z[2 * j + 1]);

		rawfile << x << "\t" << y << "\t" << z[2 * j + 0] << "\t" << z[2 * j + 1] << "\t" << ab << "\t" << r[j] << endl;

		
		if (x == XDIM - 1) rawfile << endl; 
	}

	rawfile.close();

}
/*
void plot(const string& data_dir, const string& directory, const int& time)
{

	FILE* pipe = _popen("C:/gnuplot/bin/gnuplot.exe", "w");

	if (pipe != NULL)
	{

		string data = data_dir;
		data += "/";
		data += to_string(time);
		data += "-vector.dat";

		string cilia = "Data/cilium/";

		cilia += to_string(c_num);
		cilia += "/";
		cilia += "cilia-";
		cilia += to_string(time);
		cilia += ".dat";

		string output = directory;

		output += "/";
		if (ITERATIONS > 10 && time < 10) output += "0";
		if (ITERATIONS > 100 && time < 100) output += "0";
		if (ITERATIONS > 1000 && time < 1000) output += "0";
		if (ITERATIONS > 10000 && time < 10000) output += "0";
		if (ITERATIONS > 100000 && time < 100000) output += "0";
		output += to_string(time);
		output += "-";
		output += "S";

		output += ".png";

		double t_scale = 1000.*dt*t_0;
		double t = time*t_scale;
		double x_scale = 1000000. * dx*l_0;
		double s_scale = 1000.*x_scale / t_scale;

		fprintf(pipe, "set term win\n");
		
		fprintf(pipe, "unset key\n");
		
		fprintf(pipe, "xscale = %f\n", x_scale);
		fprintf(pipe, "tscale = %f\n", t_scale);
		fprintf(pipe, "sscale = %f\n", s_scale);
		fprintf(pipe, "path1 = '%s'\n", data.c_str());
		fprintf(pipe, "path2 = '%s'\n", cilia.c_str());
		
		fprintf(pipe, "set xrange[0:%f]\n", (XDIM-1)*x_scale);
		fprintf(pipe, "set yrange[0:%f]\n", (YDIM-1)*x_scale);
		
		fprintf(pipe, "set palette rgb 33, 13, 10\n");
		fprintf(pipe, "set terminal pngcairo size %d,%d\n", 2*XDIM, 2*YDIM);
		
		fprintf(pipe, "set cbrange[0:%f]\n", SPEED*s_scale);
		
		fprintf(pipe, " set cblabel 'Fluid Speed {/Symbol m}ms^{-1}' offset 0.2,0\n");
		fprintf(pipe, "set label \"%.2fms\" at %f,%f right tc rgb \"white\" font \", 20\" front \n", t, XDIM*0.99*x_scale, YDIM*0.90*x_scale);
		fprintf(pipe, "set output \"%s\"\n", output.c_str());

		
		fprintf(pipe, "plot path1 using ($1*xscale):($2*xscale):($5*sscale) with image, path2 using (($1+%d/2)*xscale):($2*xscale) w l lc 'black'\n", LENGTH / 2 * c_num);


		fprintf(pipe, "unset output\n");
		fflush(pipe);
	}
	else puts("Could not open gnuplot\n");

	fclose(pipe);


}
*/
int main()
{
	//----------------------------INITIALISING----------------------------
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);

	cout << asctime(timeinfo) << endl;

	cout << "Initialising...\n";

	unsigned int i(0), j(0), k(0);

	int it(0);
	int last(0);

	const int size = XDIM*YDIM;

	//-------------------------------CUDA PARAMETERS DEFINITION-----------------------


	int blocksize = 500;

	int gridsize = size / blocksize;

	int blocksize2 = c_num*c_sets*LENGTH;

	int gridsize2 = 1;

	if (blocksize2 > 1000)
	{
		for (blocksize2 = 1000; blocksize2 > 0; blocksize2 -= LENGTH)
		{
			if ((c_num*LENGTH) % blocksize2 == 0)
			{
				gridsize2 = (c_num*c_sets*LENGTH) / blocksize2;
				break;
			}
		}
	}

	int xd = XDIM;

	int yd = YDIM;

	int * d_XDIM;

	int * d_YDIM;

	cudaError_t cudaStatus;

	double Q = 0.;

	//DIMENSIONS FOR DEVICE
	{
		

		cudaStatus = cudaMalloc((void**)&d_XDIM, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_YDIM, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMemcpy(d_XDIM, &xd, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_YDIM, &yd, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

	}

	//------------------------------------------ERROR------------------------------------------------


	double l_error = (l_0*dx)*(l_0*dx);
	double t_error = (t_0*dt)*(t_0*dt);
	double c_error = (t_0*dt)*(t_0*dt) / ((l_0*dx)*(l_0*dx));
	double Ma = 1.*SPEED / C_S;
	double p_runtime;


	//-------------------------------------------ASSIGN CELL VALUES ON HEAP-----------------------------

	double * u;								//VELOCITY VECTOR

	u = new double[2 * size];

	double * rho;							//DENSITY

	rho = new double[size];

	double * f0;							//EQUILIBRIUM DISTRIBUTION FUNCTION

	f0 = new double[9 * size];

	double * f;								//DISTRIBUTION FUNCTION

	f = new double[9 * size];

	double * f1;							//POST COLLISION DISTRIBUTION FUNCTION

	f1 = new double[9 * size];

	double * force;							//MACROSCOPIC BODY FORCE VECTOR

	force = new double[2 * size];

	double * F;								//LATTICE BOLTZMANN FORCE

	F = new double[9 * size];

	int Ns = LENGTH * c_num * c_sets;		//NUMBER OF BOUNDARY POINTS


	double * s;							//BOUNDARY POINTS

	double * cilium;						//DATA FROM FILE

	double inval;

	double * u_s;						//BOUNDARY POINT VELOCITY

	double * F_s;						//BOUNDARY FORCE

	double * epsilon;

	cilium = new double[5 * c_num*LENGTH];

	s = new double[2 * Ns];

	u_s = new double[2 * Ns];

	F_s = new double[2 * Ns];

	epsilon = new double[Ns];


	//----------------------------------------CREATE DEVICE VARIABLES-----------------------------

	double * d_u;								//VELOCITY VECTOR

	double * d_rho;							//DENSITY

	double * d_f0;							//EQUILIBRIUM DISTRIBUTION FUNCTION

	double * d_f;								//DISTRIBUTION FUNCTION

	double * d_f1;							//POST COLLISION DISTRIBUTION FUNCTION

	double * d_TAU;

	double * d_TAU2;

	double * d_centre;

	double * d_force;

	double * d_F;

	double * d_F_s;

	double * d_s;

	double * d_u_s;

	int * d_Ns;

	double * d_epsilon;

	int * d_it;


	double * d_Q;

	



	//---------------------------CUDA MALLOC-------------------------------------------------------------
	{
		cudaStatus = cudaMalloc((void**)&d_u, 2 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_rho, size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_f0, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_f, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_f1, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed,");
		}

		cudaStatus = cudaMalloc((void**)&d_TAU, sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_TAU2, sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_centre, 2 * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_force, 2 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_F, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_Ns, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_it, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}
		
		cudaStatus = cudaMalloc((void**)&d_Q, sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

	}

	{

		cudaStatus = cudaMalloc((void**)&d_F_s, 2 * Ns * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of F_s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_s, 2 * Ns * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_u_s, 2 * Ns * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_epsilon, Ns * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of epsilon failed!\n");
		}


		cudaStatus = cudaMemcpy(d_Ns, &Ns, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of Nsfailed!\n");
		}


	}

	//----------------------------------------DEFINE DIRECTORIES----------------------------------

	string raw_data = "Data/Raw/";
	
	raw_data += to_string(c_num);

	string img_data = "Data/Img/";
	
	img_data += to_string(c_num);


	//----------------------------------------BOUNDARY INITIALISATION------------------------------------------------

	string flux = raw_data + "/flux.dat";

	string parameters = raw_data + "/SimLog.txt";

	string input = "Data/cilium/";
	input += to_string(c_num);

	ifstream fsA(input.c_str());

	ofstream fsB(flux.c_str());

	ofstream fsC(parameters.c_str());

	fsB.open(flux.c_str(), ofstream::trunc);

	fsB.close();

	fsC.open(parameters.c_str(), ofstream::trunc);

	fsC.close();


	//----------------------------------------INITIALISE ALL CELL VALUES---------------------------------------

	for (j = 0; j < XDIM*YDIM; j++)
	{
		rho[j] = RHO_0;
		u[2 * j + 0] = 0.0;
		u[2 * j + 1] = 0.0;

		force[2 * j + 0] = 0.;
		force[2 * j + 1] = 0.;


		for (i = 0; i < 9; i++)
		{
			f0[9 * j + i] = 0.;
			f[9 * j + i] = 0.;
			f1[9 * j + i] = 0.;
			F[9 * j + i] = 0.;
		}

	}

	//------------------------------------------------------COPY INITIAL VALUES TO DEVICE-----------------------------------------------------------

	//CUDA MEMORY COPIES
	{
		cudaStatus = cudaMemcpy(d_u, u, 2 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_rho, rho, size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_f0, f0, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_f, f, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_f1, f1, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_TAU, &TAU, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_TAU2, &TAU2, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_centre, centre, 2 * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_force, force, 2 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_F, F, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_Q, &Q, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

	}

	//------------------------------------------------------SET INITIAL DISTRIBUTION TO EQUILIBRIUM-------------------------------------------------

	equilibrium << <gridsize, blocksize >> > (d_u, d_rho, d_f0, d_force, d_F, d_XDIM, d_YDIM, d_TAU);				//INITIAL EQUILIBRIUM SET

	{																										// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "first equilibrium launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaMemcpy(f0, d_f0, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(F, d_F, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}


	}

	for (j = 0; j < XDIM*YDIM; j++)
	{
		for (i = 0; i < 9; i++)
		{
			f[9 * j + i] = f0[9 * j + i];
		}
	}

	cudaStatus = cudaMemcpy(d_f, f, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of f failed!\n");
	}



	//-----------------------------------------------------OUTPUT PARAMETERS------------------------------------------------------------------------


	fsC.open(parameters.c_str(), ofstream::trunc);

	fsC.close();

	fsC.open(parameters.c_str(), ofstream::app);

	fsC << asctime(timeinfo) << endl;
	fsC << "Size: " << XDIM << "x" << YDIM << endl;
	fsC << "Iterations: " << ITERATIONS << endl;
	fsC << "Reynolds Number: " << Re << endl;
	fsC << "Relaxation times: " << TAU << ", " << TAU2 << endl;
	//if (TAU <= 0.6) fsC << "POSSIBLE INSTABILITY! Relaxation time: " << TAU << endl;
	//if (TAU >= 2.01) fsC << "POSSIBLE INACCURACY! Relaxation time: " << TAU << endl;

	fsC << "Spatial step: " << dx*l_0 << "m" << endl;
	fsC << "Time step: " << dt*t_0 << "s" << endl;
	fsC << "Mach number: " << Ma << endl;
	fsC << "Spatial discretisation error: " << l_error << endl;
	fsC << "Time discretisation error: " << t_error << endl;
	fsC << "Compressibility error: " << c_error << endl;


	fsC << "\nThreads per block: " << blocksize << endl;
	fsC << "Blocks: " << gridsize << endl;


	//--------------------------ITERATION LOOP-----------------------------
	cout << "Running Simulation...\n";

	double start = seconds();

	for (it = 0; it < ITERATIONS; it++)
	{
	
		cudaStatus = cudaMemcpy(d_it, &it, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of it failed!\n"); }

		//--------------------------IMPORT CILIUM DATA-------------------------

		int phase = (it) % T;

		if (phase == 0) phase = T;

		string input = "Data/cilium/";
		input += to_string(c_num);
		input += "/";
		input += "cilia-";
		input += to_string(phase);
		input += ".dat";

		fsA.open(input.c_str());

		j = 0;

		while (fsA >> inval)
		{
			cilium[j] = inval;

			j++;
		}


		j = 0;

		fsA.close();


		for (int n = 0; n < c_sets; n++)
		{
			

			for (j = 0; j < c_num*LENGTH; j++)
			{
				k = j + n*c_num*LENGTH;

				s[2 * k + 0] = n*LENGTH/2.*c_num + (LENGTH/2.*c_num)/2. + cilium[5 * j + 0];

				if (s[2 * k + 0] < 0) s[2 * k + 0] += XDIM;
				else if (s[2 * k + 0] > XDIM) s[2 * k + 0] -= XDIM;

				s[2 * k + 1] = cilium[5 * j + 1];

				u_s[2 * k + 0] = cilium[5 * j + 2];
				u_s[2 * k + 1] = cilium[5 * j + 3];

				epsilon[k] = cilium[5 * j + 4];
			}


			
		}



		//---------------------------CILIUM COPY---------------------------------------- 

		{

			cudaStatus = cudaMemcpy(d_epsilon, epsilon, Ns * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of epsilon failed!\n");
			}

			cudaStatus = cudaMemcpy(d_s, s, 2 * Ns * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of s failed!\n");
			}

			cudaStatus = cudaMemcpy(d_u_s, u_s, 2 * Ns * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u_s failed!\n");
			}

			cudaStatus = cudaMemcpy(d_F_s, F_s, 2 * Ns * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of F_s failed!\n");
			}
		}


		//---------------------------IMMERSED BOUNDARY LATTICE BOLTZMANN STEPS-------------------

		

		equilibrium << <gridsize, blocksize >> > (d_u, d_rho, d_f0, d_force, d_F, d_XDIM, d_YDIM, d_TAU);				//EQUILIBRIUM STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "equilibrium launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		collision << <gridsize, blocksize >> > (d_f0, d_f, d_f1, d_F, d_TAU, d_TAU2, d_XDIM, d_YDIM, d_it);						//COLLISION STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		streaming << <gridsize, blocksize >> > (d_f1, d_f, d_XDIM, d_YDIM);												//STREAMING STEP

		{																											// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}

		}

		macro << <gridsize, blocksize >> > (d_f, d_u, d_rho, d_XDIM, d_YDIM);											//MACRO STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		interpolate << <gridsize2, blocksize2 >> > (d_rho, d_u, d_Ns, d_u_s, d_F_s, d_s, d_XDIM);											//IB INTERPOLATION STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "interpolate launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		spread << <gridsize, blocksize >> > (d_rho, d_u, d_f, d_Ns, d_u_s, d_F_s, d_force, d_s, d_XDIM, d_Q);	//IB SPREADING STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "spread launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}

			cudaStatus = cudaMemcpy(rho, d_rho, size * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of rho failed!\n");
			}

			cudaStatus = cudaMemcpy(u, d_u, 2 * size * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u failed!\n");
			}

			cudaStatus = cudaMemcpy(&Q, d_Q, sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}


		}

		//----------------------------DATA OUTPUT------------------------------
		if (it % INTERVAL == 0)
		{
			last = it - INTERVAL;

			print(rho, u, raw_data, it);
			

			if (last >= INTERVAL)
			{
				//plot(raw_data, img_data, last);
				
			}

			fsB.open(flux.c_str(), ofstream::app);

			fsB << it*1000.*dt*t_0 << "\t" << Q*1000000. * dx*l_0*1000000. * dx*l_0 << endl;

			fsB.close();
		}


		if (it == INTERVAL)
		{
			double cycle = seconds();

			p_runtime = (cycle - start)*(ITERATIONS / INTERVAL);

			time_t p_end = rawtime + p_runtime;

			timeinfo = localtime(&p_end);

			int hours(0), mins(0);
			double secs(0.);

			if (p_runtime > 3600) hours = nearbyint(p_runtime / 3600 - 0.5);
			if (p_runtime > 60) mins = nearbyint((p_runtime - hours * 3600) / 60 - 0.5);
			secs = p_runtime - hours * 3600 - mins * 60;

			cout << "\nProjected runtime: ";
			if (hours < 10) cout << 0;
			cout << hours << ":";
			if (mins < 10) cout << 0;
			cout << mins << ":";
			if (secs < 10) cout << 0;
			cout << fixed << setprecision(2) << secs;

			cout << "\nCompletion time: " << asctime(timeinfo) << endl;

			fsC << "\nCompletion time: " << asctime(timeinfo) << endl;

			fsC.close();
		}
	}

	fsB.open(flux.c_str(), ofstream::app);

	fsB << it*1000.*dt*t_0 << "\t" << Q*1000000. * dx*l_0*1000000. * dx*l_0 << endl;

	fsB.close();
	
	double end = seconds();

	double runtime = end - start;

	int hours(0), mins(0);
	double secs(0.);

	if (runtime > 3600) hours = nearbyint(runtime / 3600 - 0.5);
	if (runtime > 60) mins = nearbyint((runtime - hours * 3600) / 60 - 0.5);
	secs = runtime - hours * 3600 - mins * 60;

	fsC.open(parameters.c_str(), ofstream::app);

	fsC << "\nTotal runtime: ";
	if (hours < 10) fsC << 0;
	fsC << hours << ":";
	if (mins < 10) fsC << 0;
	fsC << mins << ":";
	if (secs < 10) fsC << 0;
	fsC << secs << endl;
	fsC << "Net Q = " << Q << " Avg Q = " << Q / 1.*(it / T) << endl;

	fsC.close();


	return 0;
}