#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
//#include "defs.h"
#include "projectedGradientDescent.h"
//declaring the global parameters for em with smoothed l0
/*
GLOBAL_PARAMETER(float, ARMIJO_BETA,"armijo_beta","pgd optimization parameter beta used in armijo line search",PARAM_SMOOTHED_LO,0.1);
GLOBAL_PARAMETER(float, ARMIJO_SIGMA,"armijo_sigma","pgd optimization parameter sigma used in armijo line search",PARAM_SMOOTHED_LO,0.0001);
GLOBAL_PARAMETER(float, L0_BETA,"smoothed_l0_beta","optimiztion parameter beta that controls the sharpness of the smoothed l0 prior",PARAM_SMOOTHED_LO,0.5);
GLOBAL_PARAMETER(float, L0_ALPHA,"smoothed_l0_alpha","optimization parameter beta that controls the sharpness of the smoothed l0 prior",PARAM_SMOOTHED_LO,0.0);
*/


using namespace std;

/*
void inline printVector(vector<float> & v)
{
	for (unsigned int i = 0; i< v.size();i++)
	{
		cout<<v[i]<<" ";
	}
	cout<<endl;
}
*/
//template <class PROB>
//bool descending (float i,float j) { return (i>j); }

//typedef float PROB ;
//typedef float COUNT ;


/*
void projectOntoSimplex(vector<float> &v,vector<float> &projection)
{
	vector<float> mu_vector (v.begin(),v.end());
	sort(mu_vector.begin(),mu_vector.end(),descending);
	//vector<float>  projection_step;		
	vector<float> sum_vector (mu_vector.size(),0.0);
	float sum = 0;
	//float max = -99999999;
	vector<float>::iterator it;
	int counter = 1;
	int max_index  = -1;
	float max_index_sum = -99999999;
	for (it = mu_vector.begin() ;it != mu_vector.end(); it++)
	{
		sum += *it;
		float temp_rho = *it - (1/(float)counter)*(sum-1);
		if (temp_rho > 0)
		{
			max_index = counter;
			max_index_sum = sum;
		}
		counter++;
	}
	
	float theta = 1/(float) max_index *(max_index_sum -1);
	//vector <float> projection ;
	
	for (it = v.begin() ;it != v.end(); it++)
	{
		float value = max((*it)-theta,(float)0.0);
		projection.push_back(value);
//		cout<<value<<endl;
	}
	//cout<<"the size of final ector is "<<projection.size()<<endl;
	return;
}
*/


//evaluates the value of the function and returns the function value
inline float evalFunctionDD(const vector<float> & expected_counts ,
    vector<float> & current_point,
    const vector<float> &lagrange_lambdas,
    int lagrange_sign,
    float unigram_prob,
    float reg_lambda)
{
  //cerr<<"Current point is "<<current_point<<endl;
 	int num_elements = expected_counts.size();
	float func_value = 0.0;
	for (int i =0 ;i<num_elements;i++)
	{
		if (current_point[i] == 0.0 && expected_counts[i] != 0.0)
		{
			cout<<"the probability in position "<<i<<" was 0"<<" and the fractional count was not 0"<<endl;
			exit(0);
		}
		if (current_point[i] != 0.0)
		{
			func_value += expected_counts[i] * log(current_point[i]) +  
        lagrange_sign*lagrange_lambdas[i]*current_point[i]*unigram_prob*reg_lambda;//+ L0_ALPHA * exp(-current_point[i]/L0_BETA);
      //cerr<<"the func value is "<<func_value<<endl;
		}
		else
		{
      cerr<<"The current point "<<i<<" was 0"<<endl;
      exit(0);
			//func_value += L0_ALPHA * exp(-current_point[i]/L0_BETA);
		}
	}
	
	return(-func_value);
}

//template <class COUNT,class PROB>
void inline evalGradientDD(const vector<float> & expected_counts,
    vector<float> & current_point,
    vector<float> & gradient,
    const vector<float> &lagrange_lambdas,
    int lagrange_sign,
    float unigram_prob,
    float reg_lambda)
{
	int num_elements = expected_counts.size();
	for (int i =0 ;i<num_elements;i++)
	{
		if (current_point[i] == 0 && expected_counts[i] != 0.0 )
		{
			cout<<"the probability in position "<<i<<" was 0"<<" and the fractional count was not 0"<<endl;
			exit(0);
		}
		if (current_point[i] != 0.0)
		{
			gradient[i] = -1 *(expected_counts[i] / current_point[i] +
          lagrange_sign*lagrange_lambdas[i]*unigram_prob*reg_lambda);//- L0_ALPHA * exp(-current_point[i]/L0_BETA)/L0_BETA	);
		}
		else
		{
      cerr<<"The current point "<<i<<" was 0"<<endl;
      exit(0);
			//gradient[i] = -1 *(- L0_ALPHA * exp(-current_point[i]/L0_BETA)/L0_BETA	);
		}
		//printf ("the gradient is %.16f\n",gradient[i]);
	}
}


//template <class COUNT,class float>
void projectedGradientDescentWithArmijoRuleDualDecomp(const vector<float> & expected_counts ,
    const vector<float> & current_prob,
    vector<float> & new_prob,
    vector<float> & lagrange_lambdas,
    int lagrange_sign,
    float unigram_prob,
    float reg_lambda)
{
  cerr<<"Reg lambda is "<<reg_lambda<<endl;
	//cout<<" we are in projected grad"<<endl;
	int num_elements = expected_counts.size();
	//cout<<"projected gradient descent here"<<endl;
	vector<float> current_point(current_prob);
	//cout <<"the number of PGD iterations is "<<NUM_PGD_ITERATIONS<<endl;
  //EVALUATING THE FUNCTION AT THE CURRENT POINT
  cerr<<"The lagrange lambdas are "<<lagrange_lambdas<<endl;
  getchar();
  cerr<<"the expected counts are "<<expected_counts<<endl;
  getchar();
  cerr<<"Current point is "<<current_prob<<endl;
  getchar();
  float current_function_value = evalFunctionDD(expected_counts,
      current_point,
      lagrange_lambdas,
      lagrange_sign,
      unigram_prob,
      reg_lambda);
  cerr<<"Before starting dd pgd, the function value was "<<current_function_value<<endl;
	for (int time = 1; time <= NUM_PGD_ITERATIONS; time++)
	{
		cerr<<"dd iteration is"<<time<<endl;

		//getchar();
		vector<float> gradient (num_elements,(float) 0.0);
		evalGradientDD(expected_counts,
        current_point,
        gradient,
        lagrange_lambdas,
        lagrange_sign,
        unigram_prob,
        reg_lambda);
    //cerr<<"Computed the gradient"<<endl;
    //cerr<<"The gradient was "<<gradient<<endl;
    //getchar();
		vector<float> new_point (num_elements,(float) 0.0);
		//moving in the opposite direction of the gradient
		for (int i =0 ;i<num_elements;i++)
		{
			new_point[i] = current_point[i] - ETA * gradient[i];	
		}
    //cerr<<"The gradient is "<<gradient<<endl;
		vector<float> new_feasible_point(num_elements,0.0) ;//(num_elements,(float) 0.0);
		// now projecting on the simplex
    //cerr<<"We are projecting "<<new_point<<" onto the simplex"<<endl;
		projectOntoSimplex(new_point,new_feasible_point);
    //cerr<<"We projected onto simplex"<<endl;
    //cerr<<"new feasible point is "<<new_feasible_point<<endl;
		//printVector(new_feasible_point);
		//cout<<"feasible point dimension is "<<new_feasible_point.size()<<endl;
		//int num_zero_entries = 0;
		/*
		for (int i = 0;i<new_feasible_point.size();i++)
		{
			if (new_feasible_point[i] == 0)
			{
				num_zero_entries++ ;
			}
		}
		*/
		//cout<<"the number of zero entries is "<<num_zero_entries<<endl;
		//getchar();
		//cout<<"armijo beta is "<<ARMIJO_BETA<<endl;
		//cout<<"armijo sigma is "<<ARMIJO_SIGMA<<endl;
		float armijo_bound = 0.0;
		for (int i =0 ;i<num_elements;i++)
		{
			float bound_term = ARMIJO_SIGMA * ARMIJO_BETA * gradient[i] * (new_feasible_point[i] - current_point[i]); 	
			//cout<<"the grad is "<<gradient[i]<<" the new feasible point is "<<new_feasible_point[i]<<" current point is "<<current_point[i]<<endl;
			//cout<<"the bound term is "<<bound_term<<endl;
			armijo_bound -= bound_term;
			//cout<<"temp armijo bound "<<armijo_bound<<endl;
		}
		
		//getchar();
		bool terminate_line_srch = 0;
		int num_steps = 1;
		float current_alpha = ARMIJO_BETA ;
		float final_alpha = 0.0 ; //if the function value does not improve at all, then the armijo beta should be 1
		float current_armijo_bound = armijo_bound;
		float best_func_value = current_function_value;
		bool no_update = 1;
		//cout<<"current function value is "<<current_function_value<<endl;
		//printf ("current function value is %.15f\n",current_function_value);
		while(terminate_line_srch != 1 && num_steps <= 20)
		{	
			//cout<<"current armijo bound is "<<current_armijo_bound<<endl;
			//cout<<"we are in the while loop"<<endl;
			//cout<<"num steps is "<<num_steps<<endl;
		//	current_beta = 
			vector<float> temp_point (num_elements,(float) 0.0);
			for (int i =0 ;i<num_elements;i++)		
			{
				temp_point[i] = (1.0 - current_alpha) * current_point[i] + current_alpha * new_feasible_point[i];
			}
      //cerr<<"temp point is "<<temp_point<<endl;
			float func_value_at_temp_point = evalFunctionDD(expected_counts,
          temp_point,
          lagrange_lambdas,
          lagrange_sign,
          unigram_prob,
          reg_lambda);
			//cerr<<"function value at temp point is "<<func_value_at_temp_point<<endl;
			//printf ("function value at temp point is %.15f and the iteration number is %d \n",func_value_at_temp_point,num_steps);
			//printf ("current alpha is %.15f\n",current_alpha);
			//getchar();
			if (func_value_at_temp_point < best_func_value)
			{
				best_func_value = func_value_at_temp_point;
				final_alpha = current_alpha;
				no_update = 0;
				//cout<<"we arrived at a better function value"<<endl;
				//getchar();
			}
			
			if (current_function_value - func_value_at_temp_point >= current_armijo_bound)
			{
				//cout<<"the terminate line src condition was met "<<endl;
				terminate_line_srch = 1;
			}
			current_alpha *= ARMIJO_BETA;
			current_armijo_bound *= ARMIJO_BETA;
			num_steps += 1;
			//getchar();
		}
		//printf ("final alpha was %f\n",final_alpha);
		//cout<<"the value of not update was "<<no_update<<endl;
		//getchar();
	
		//vector<float> next_point ;
		if (no_update == 0)
		{
			//next_point.resize(num_elements);
			for (int i =0 ;i<num_elements;i++)
			{
				float coordinate_point = (1.0 - final_alpha)*current_point[i] + final_alpha * new_feasible_point[i];
				//next_point.push_back(coordinate_point);
				current_point[i] = coordinate_point;
			}
      current_function_value = best_func_value;
			//current_point = next_point;
		}
		else
		{
			cerr<<" not update was true"<<endl;
			break;
		}
    cerr<<"The best function value is "<<best_func_value<<endl;
	}
	new_prob = current_point;
  //cerr<<"The new prob that we are returning is "<<new_prob<<endl;
  
}


