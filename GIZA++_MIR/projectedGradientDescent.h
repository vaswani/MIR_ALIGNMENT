#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
//#include "defs.h"
#include <omp.h>
//#include "dualDecomp.h"

//#include "vocab.h"

//declaring the global parameters for em with smoothed l0
/*
GLOBAL_PARAMETER(float, ARMIJO_BETA,"armijo_beta","pgd optimization parameter beta used in armijo line search",PARAM_SMOOTHED_LO,0.1);
GLOBAL_PARAMETER(float, ARMIJO_SIGMA,"armijo_sigma","pgd optimization parameter sigma used in armijo line search",PARAM_SMOOTHED_LO,0.0001);
GLOBAL_PARAMETER(float, L0_BETA,"smoothed_l0_beta","optimiztion parameter beta that controls the sharpness of the smoothed l0 prior",PARAM_SMOOTHED_LO,0.5);
GLOBAL_PARAMETER(float, L0_ALPHA,"smoothed_l0_alpha","optimization parameter beta that controls the sharpness of the smoothed l0 prior",PARAM_SMOOTHED_LO,0.0);
*/


using namespace std;

void inline printVector(vector<float> & v)
{
	for (unsigned int i = 0; i< v.size();i++)
	{
		cout<<v[i]<<" ";
	}
	cout<<endl;
}

//template <class PROB>
bool descending (float i,float j) { return (i>j); }

//typedef float PROB ;
//typedef float COUNT ;



void projectOntoSimplex(
    vector<float> &v,
    vector<float> &projection)
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
  //cerr<<" the projection is "<<projection<<endl; 
  int index = 0;
	for (it = v.begin() ;it != v.end(); it++)
	{
		float value = max((*it)-theta,(float)0.0);
		projection[index] = value;
    index++;
//		cout<<value<<endl;
	}
	//cout<<"the size of final ector is "<<projection.size()<<endl;
	return;
}

void projectPointsOntoSimplex(
    vector<vector<float> > &new_points,
    vector<vector<float> > &new_feasible_points) {
  int num_conditionals = new_feasible_points.size();
  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate(num_conditionals) 
  for( int i=2; i<num_conditionals;i++) {
    //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
    if (new_points[i].size() > 0) {
      //cout<<"projecting point "<<i<<endl;
      //cout<<"projecting a point with size "<<new_points[i].size()<<endl;
      projectOntoSimplex(new_points[i],new_feasible_points[i]);
    }
  }
  //cout<<"We finished projecting points"<<endl; 
}

float regFuncValue(
       const vector<vector<float> > &  ef_current_points,
       const vector<vector<float> > &  fe_current_points,
       const vector<float> &rowwiseExpCntsSum,
       const vector<vector<pair<unsigned int,unsigned int> > > &ef_map,
       float reg_lambda,
       vcbList &eTrainVcbList,
       vcbList &fTrainVcbList,
       regularization_type reg_option,
       regularization_func_type reg_func_option) {
 	int num_conditionals = ef_current_points.size();
	float func_value = 0.0;

  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate (num_conditionals) reduction(+ : func_value ) 
	for (int i =2 ;i<num_conditionals;i++)
	{ 
    float conditional_func_value = 0.;
    if (rowwiseExpCntsSum[i] == 0.) {
      continue;
    }
	//cerr<<"The row is "<<i<<endl;
    for (unsigned int j=0; j<ef_map[i].size(); j++) {
	  //cerr<<"the item is "<<j<<endl;
      unsigned int f_position =  ef_map[i][j].first;
      unsigned int e_position_in_f_row = ef_map[i][j].second;
	  //cerr<<"the f position is "<<f_position<<endl;
	  //cerr<<"the eposition in frow is "<<e_position_in_f_row<<endl;
      float diff = 0.;
      if (reg_option == CONDITIONAL) {
        diff = ef_current_points[i][j] - 
                        fe_current_points[f_position][e_position_in_f_row];
        if (reg_func_option == L2) {
          conditional_func_value += diff*diff;
        }
        if (reg_func_option == L1) {
          conditional_func_value += abs(diff);
        }
        if (reg_func_option == GLASSO) {
          conditional_func_value += sqrt(ef_current_points[i][j]*ef_current_points[i][j] +
            fe_current_points[f_position][e_position_in_f_row]*fe_current_points[f_position][e_position_in_f_row]);
        }

      }
      if (reg_option == JOINT) {
	    //cerr<<ef_current_points[i][j]<<endl;
	    //cerr<<fe_current_points[f_position][e_position_in_f_row]<<endl;
		//cerr<<eTrainVcbList.getProbForWord(i)<<endl;
		//cerr<<fTrainVcbList.getProbForWord(f_position)<<endl;
    //     float e_prob = eTrainVcbList.getProbForWord(f_position);
    // float f_prob = fTrainVcbList.getProbForWord(i);
        float f_prob = fTrainVcbList.getProbForWord(i);
        float e_prob = eTrainVcbList.getProbForWord(f_position);
        float ef_joint = ef_current_points[i][j]*f_prob;
        float fe_joint = fe_current_points[f_position][e_position_in_f_row]*e_prob;
        diff = ef_current_points[i][j]*f_prob -
            fe_current_points[f_position][e_position_in_f_row]*e_prob;
        // ADDING SCALING
        diff = (ef_current_points[i][j]*fTrainVcbList.getProbForWord(i) -
            fe_current_points[f_position][e_position_in_f_row]*eTrainVcbList.getProbForWord(f_position)); 
        if (reg_func_option == L2) {
          //float e_prob_sqr = e_prob*e_prob;
          //float f_prob_sqr = f_prob*f_prob;
          conditional_func_value += diff*diff;//*(e_prob_sqr+f_prob_sqr)/(e_prob_sqr*f_prob_sqr);
        }
        if (reg_func_option == L1) {
          conditional_func_value += abs(diff);//*(e_prob+f_prob)/(e_prob*f_prob);
        }
        if (reg_option == GLASSO) {
          conditional_func_value += sqrt(ef_joint*ef_joint +
            fe_joint*fe_joint);
        }
		//cerr<<"diff is "<<diff<<endl;
      }
      if (reg_option == PRODUCT) {
        conditional_func_value -= ef_current_points[i][j]*
        fe_current_points[f_position][e_position_in_f_row];
      }
      if (reg_option == PRODUCT_SQRT) {
        conditional_func_value -= sqrt(ef_current_points[i][j]*
          fe_current_points[f_position][e_position_in_f_row]);
      }
    }

    func_value += conditional_func_value;
  }
  /*
  if (reg_func_option == L2) {
    cerr<<"The unweighted l2 reg function value is "<<func_value<<endl;
  }
  if (reg_func_option == L1) {
    cerr<<"The unweighted l1 reg function value is "<<func_value<<endl;
  }
  */
  //cout<<"reg_lambda was "<<reg_lambda<<endl;
  return (reg_lambda*func_value);
   
}
float parallelExpectedLLCompute(
    const vector<vector<float> > &expected_counts,
    const vector<vector<float> > &current_points,
    const vector<float> &rowwiseExpCntsSum,
    const float lambda) {

 	int num_conditionals = expected_counts.size();
	float func_value = 0.0;

  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate (num_conditionals) reduction(+ : func_value ) 
	for (int i =2 ;i<num_conditionals;i++)
	{
    // NO NEED TO PERFORM COMPUTATION IF THE EXPECTED COUNTS SUM WAS
    // ZERO
    if (rowwiseExpCntsSum[i] == 0.) {
      continue;
    }
    float conditional_function_value = 0.;
    //cout<<"The size of the row "<<i<<" was "<<current_points[i].size()<<endl;
    for (unsigned int j=0; j<current_points[i].size(); j++) {
      if (current_points[i][j] == 0.0 && expected_counts[i][j] != 0.0)
      {
        cerr<<"the probability in position "<<j<<" was 0"<<" and the fractional count was not 0"<<endl;
        exit(0);
      }
      if (current_points[i][j] != 0.0 and expected_counts[i][j] != 0)
      {
        //cout<<"current point is "<<current_points[i][j]<<endl;
        conditional_function_value += expected_counts[i][j] * log(current_points[i][j]);// + L0_ALPHA * exp(-current_point[i]/L0_BETA);
      }
      /*
      else
      {
        func_value += L0_ALPHA * exp(-current_points[i][j]/L0_BETA);
      }
      */
    }
    func_value += conditional_function_value;
	}
	return(func_value*lambda);

}

// COMPUTING THE FUNCION VALUE AT THE CURRENT POINT. WE ARE PERFORMING
// PROJECTED GRADIENT DESCENT ON THE NEGATIVE OF THE FUNCTION VALUE
// i.e PROJECTED GRADIENT ASCENT
void evalFunction(
    const vector<vector<float> > & ef_expected_counts,
    const vector<vector<float> > & ef_current_points,
    const vector<float> & ef_rowwiseExpCntsSum,
    const vector<vector<float> > & fe_expected_counts,
    const vector<vector<float> > & fe_current_points,
    const vector<float> & fe_rowwiseExpCntsSum,
    const float lambda_ef,
    const float lambda_fe,
    const float reg_lambda,
    const vector<vector<pair<unsigned int,unsigned int> > > &ef_map,
    vcbList &eTrainVcbList,
    vcbList &fTrainVcbList,
    regularization_type reg_option,
    regularization_func_type reg_func_option,
    float *exp_complete_ll_term_value,
    float *reg_term_value)
{
  //cout<<"In eval function"<<endl;
  //float func_value = 0.;
  //float reg_func_value=0;
  // FIRST COMPUTING THE EXPECTED COMPLETE DATA LOG LIKELIHOOD IN e given f directiona
  //cerr<<"Computing expected complete data LL in the e|f direction"<<endl;
  (*exp_complete_ll_term_value) -= parallelExpectedLLCompute(
    ef_expected_counts,
    ef_current_points,
    ef_rowwiseExpCntsSum,
    lambda_ef);

  //cerr<<"Computing expected complete data LL in the f|e direction"<<endl;
  (*exp_complete_ll_term_value) -= parallelExpectedLLCompute(
    fe_expected_counts,
    fe_current_points,
    fe_rowwiseExpCntsSum,
    lambda_fe);
  // Now computing the regularization constant. 
  // For now, the L2 norm
   //cerr<<"Computing the function value for the l2 term"<<endl;
   (*reg_term_value) += regFuncValue(
       ef_current_points,
       fe_current_points,
       ef_rowwiseExpCntsSum,
       ef_map,
       reg_lambda,
       eTrainVcbList,
       fTrainVcbList,
       reg_option,
       reg_func_option);
   //func_value += reg_func_value;
   /*
   if (reg_func_option == L1) {
    cerr<<"The l1 reg func value was "<<reg_func_value<<endl;
   }
   if (reg_func_option == L2) {
    cerr<<"The l2 reg func value was "<<reg_func_value<<endl;
   }
  */
	//return(func_value);
  return;
}

void regGradient(
      const vector<vector<float> > &ef_current_points,
      vector<vector<float> > &ef_gradients,
      const vector<vector<float> > &fe_current_points,
      vector<vector<float> > &fe_gradients,
      const float reg_lambda,
      const vector<vector<pair<unsigned int,unsigned int> > > &ef_map,
      vcbList &eTrainVcbList,
      vcbList &fTrainVcbList,
      regularization_type reg_option,
      regularization_func_type reg_func_option) {
  
  //First computing the gradient in the ef direction
  int num_conditionals = ef_current_points.size();

  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate(num_conditionals)
  for(int i=2; i<num_conditionals; i++) {
    for (unsigned int j=0; j<ef_map[i].size(); j++) {
      unsigned int f_position =  ef_map[i][j].first;
      unsigned int e_position_in_f_row = ef_map[i][j].second;
      float e_prob = eTrainVcbList.getProbForWord(f_position);
      float f_prob = fTrainVcbList.getProbForWord(i);
      float diff = 0;
      if (reg_option == CONDITIONAL) {
        diff = ef_current_points[i][j] - 
                        fe_current_points[f_position][e_position_in_f_row];
      }
      if (reg_option == JOINT) {
        diff = ef_current_points[i][j]*f_prob -
            fe_current_points[f_position][e_position_in_f_row]*e_prob;
      }
      float fe_gradient_value =0;
      float ef_gradient_value =0;
      if (reg_func_option == L2) {
        if (reg_option == CONDITIONAL) {
          float gradient_value = 2*diff;
          ef_gradient_value = gradient_value;
          fe_gradient_value = -gradient_value;
        }
        if (reg_option == JOINT) {
          //float e_prob_sqr = e_prob*e_prob;
          //float f_prob_sqr = f_prob*f_prob;
          //float scale = (e_prob_sqr+f_prob_sqr)/(e_prob_sqr*f_prob_sqr);
          ef_gradient_value = 2*diff;//*f_prob*scale;
          fe_gradient_value = -2*diff;//*e_prob*scale;
        }
      }

      if (reg_func_option == L1) {
        //float scale = (e_prob+f_prob)/(e_prob*f_prob);
        if (diff > 0) {
          if (reg_option == CONDITIONAL) {
          ef_gradient_value = 1;
          fe_gradient_value = -1;
          }
          if (reg_option == JOINT) {
            ef_gradient_value = f_prob;//*scale;
            fe_gradient_value = -e_prob;//*scale;
          }
        }
        if (diff < 0) {
          if (reg_option == CONDITIONAL) {
            ef_gradient_value = -1;
            fe_gradient_value = 1;
          }
          if (reg_option == JOINT) {
            ef_gradient_value = -f_prob;//*scale;
            fe_gradient_value = e_prob;//*scale;
          }
        }
      }
      if (reg_func_option == GLASSO) {
        if (reg_option == CONDITIONAL) {
          float denom = sqrt(ef_current_points[i][j]*ef_current_points[i][j] + 
            fe_current_points[f_position][e_position_in_f_row]*fe_current_points[f_position][e_position_in_f_row]);
          ef_gradient_value =  ef_current_points[i][j]/denom;
          fe_gradient_value = fe_current_points[f_position][e_position_in_f_row]/denom;
        }
        else if (reg_option == JOINT) {
          float e_joint = ef_current_points[i][j]*e_prob;
          float f_joint = fe_current_points[f_position][e_position_in_f_row]*f_prob;
          float denom = sqrt(e_joint*e_joint + f_joint*f_joint); 
          ef_gradient_value =  e_joint/denom;
          fe_gradient_value = f_joint/denom;
        }
      }

      if (reg_option == PRODUCT) {
        ef_gradient_value = -fe_current_points[f_position][e_position_in_f_row];
        fe_gradient_value = -ef_current_points[i][j];
      }

      if (reg_option == PRODUCT_SQRT) {
        float denom = 2*sqrt(ef_current_points[i][j]*fe_current_points[f_position][e_position_in_f_row]);
        ef_gradient_value = -fe_current_points[f_position][e_position_in_f_row]/denom;
        fe_gradient_value = -ef_current_points[i][j]/denom;
      }

      //cerr<<"ef gradient value is "<<ef_gradient_value<<endl;
      //cerr<<"fe gradient value is "<<fe_gradient_value<<endl;
      ef_gradients[i][j] += reg_lambda*ef_gradient_value;
      fe_gradients[f_position][e_position_in_f_row] += reg_lambda*fe_gradient_value;
    }
  }
}

void evalGradientFromExpLogLikelihood(
    const vector<vector<float> > &expected_counts,
    const vector<float> &rowwiseExpCntsSum,
    const vector<vector<float> > &current_points,
    vector<vector<float> > &gradients,
    const float lambda) {
  //First evaluating the gradients from the expected complete data log likeliood
	int num_conditionals = expected_counts.size();

  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate(num_conditionals)
	for (int i =2; i<num_conditionals; i++)
	{
    if (rowwiseExpCntsSum[i] == 0) {
      continue;
    }
    for (unsigned int j=0; j<expected_counts[i].size(); j++) {
      if (current_points[i][j] == 0 && expected_counts[i][j] != 0.0 )
      {
        cerr<<"the probability in position "<<j<<" was 0"<<" and the fractional count was not 0"<<endl;
        exit(0);
      }
      if (current_points[i][j] != 0.0)
      {
        gradients[i][j] = -1 *lambda*(expected_counts[i][j] / current_points[i][j]) ; //- L0_ALPHA * exp(-current_point[i]/L0_BETA)/L0_BETA	);
      }
    }
  }
}

//template <class COUNT,class PROB>
void inline evalGradient(
      const vector<vector<float> > &ef_expected_counts,
      const vector<vector<float> > &ef_current_points,
      const vector<float> &ef_rowwiseExpCntsSum,
      vector<vector<float> > &ef_gradients,
      const vector<vector<float> > &fe_expected_counts,
      const vector<vector<float> > &fe_current_points,
      const vector<float> &fe_rowwiseExpCntsSum,
      vector<vector<float> > &fe_gradients,
      float lambda_ef,
      float lambda_fe,
      float reg_lambda,
      const vector<vector<pair<unsigned int,unsigned int> > > &ef_map,
      vcbList &eTrainVcbList,
      vcbList &fTrainVcbList,
      regularization_type reg_option,
      regularization_func_type reg_func_option) {

  evalGradientFromExpLogLikelihood(
      ef_expected_counts,
      ef_rowwiseExpCntsSum,
      ef_current_points,
      ef_gradients,
      lambda_ef);
  evalGradientFromExpLogLikelihood(
      fe_expected_counts,
      fe_rowwiseExpCntsSum,
      fe_current_points,
      fe_gradients,
      lambda_fe);

  regGradient(
      ef_current_points,
      ef_gradients,
      fe_current_points,
      fe_gradients,
      reg_lambda,
      ef_map,
      eTrainVcbList,
      fTrainVcbList,
      reg_option,
      reg_func_option);
}

void zeroInitVectorOfVector(
    const vector<vector<float> > &source,
    vector<vector<float> > &target) {
  int num_conditionals = source.size();
  //#pragma omp parallel for firstprivate(num_conditionals)
  for (int i=0; i<num_conditionals; i++) {
    int target_row_size = source[i].size();
    target[i] = vector<float> (target_row_size,0.0);
  }
}

//Initializing the gradients to zero vectors
void initGradientsToZero(
        vector<vector<float> > &ef_gradients,
        const vector<vector<float> > &ef_current_points,
        vector<vector<float> > &fe_gradients,
        const vector<vector<float> > &fe_current_points) {
  zeroInitVectorOfVector(ef_current_points,ef_gradients);
  zeroInitVectorOfVector(fe_current_points,fe_gradients);
}

void getSingleInterpolatedPoints(
    vector<vector<float> > &new_feasible_points,
    vector<vector<float> > &current_points,
    vector<vector<float> > &temp_points,
    float current_alpha)  {
  int num_conditionals = new_feasible_points.size();

  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate(num_conditionals)
  for (int i=2; i<num_conditionals; i++) {
    for (unsigned int j=0; j<new_feasible_points[i].size(); j++) {
      temp_points[i][j] = (1.0 - current_alpha) * current_points[i][j] + current_alpha * new_feasible_points[i][j];
    }
  }

}

void getInterpolatedPoints(
    vector<vector<float> > &ef_new_feasible_points,
    vector<vector<float> > &ef_current_points,
    vector<vector<float> > &ef_temp_points,
    vector<vector<float> > &fe_new_feasible_points,
    vector<vector<float> > &fe_current_points,
    vector<vector<float> > &fe_temp_points,
    float current_alpha) {

  getSingleInterpolatedPoints(ef_new_feasible_points,
      ef_current_points,
      ef_temp_points,
      current_alpha);

  getSingleInterpolatedPoints(fe_new_feasible_points,
      fe_current_points,
      fe_temp_points,
      current_alpha);

}
void singleGradientStep(
    vector<vector<float> > &new_points,
    const vector<vector<float> > &current_points,
    const vector<vector<float> > &gradients,
    float eta) {
  int num_conditionals = new_points.size();

  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate(num_conditionals)
  for (int i=2; i<num_conditionals; i++) {
    for (unsigned int j=0; j<new_points[i].size(); j++) {
      new_points[i][j] = current_points[i][j] - eta*gradients[i][j];
    }
  }
}

void takeGradientStep(
    vector<vector<float> > &ef_new_points,
    const vector<vector<float> > &ef_current_points,
    const vector<vector<float> > &ef_gradients,
    vector<vector<float> > &fe_new_points,
    const vector<vector<float> > &fe_current_points,
    const vector<vector<float> > &fe_gradients,
    float eta) {
  //Move the probabilities in the direction of the gradient
  singleGradientStep(
      ef_new_points,
      ef_current_points,
      ef_gradients,
      eta);
  singleGradientStep(
      fe_new_points,
      fe_current_points,
      fe_gradients,
      eta);
}

void reNormalize(
    const vector<float> &expected_counts,
    const float expected_counts_sum,
    vector<float> &new_point) {
  if (expected_counts_sum == 0 ) {
    // assign uniform probabilities
    float unif_prob = 1.0/new_point.size();
    for (unsigned int i=0;i<new_point.size();i++) {
      new_point[i] = unif_prob;
    }
  } else {
    for (unsigned int i=0;i<new_point.size();i++) {
      new_point[i] = expected_counts[i]/expected_counts_sum;
    }
  }
  return;
}

//PERFORM EXPONENTIATED GRAIDENT UPDATES FOR A SINGLE 
//PROBABILITY DISTRIBUTION
void expGradientUpdateSingleGroup(
    vector<float> &current_point,
    vector<float> &new_point,
    vector<float> &gradient,
    float step_size) {
  float gradient_norm = 0.;
  if (current_point.size() != new_point.size() || 
      current_point.size() != gradient.size() ){
    cerr<<"WARNING: in exp gradient update, the current point size was "<<
      "not equal to the gradient size"<<endl;
  }
  vector<float> gradient_terms(gradient.size(),0.0);
  // FIRST COMPUTE THE GRADIENT NORMALIZER
  for (unsigned int i=0; i<current_point.size(); i++){
    //Max possible value of the term is 20 and min possible value of the term is -20
    float exp_inner_term = -step_size*gradient[i];
    if (exp_inner_term > 20.0){
      exp_inner_term = 20.0;
    }
    else if (exp_inner_term < -20.0){
      exp_inner_term = -20.0;
    }
    gradient_terms[i] = current_point[i]*exp(exp_inner_term);
    gradient_norm += gradient_terms[i];
  }
  //cerr<<"the gradient was "<<gradient<<endl;
  //cerr<<"gradient terms were "<<gradient_terms<<endl;
  //cerr<<"the gradient norm was "<<gradient_norm<<endl;
  for (unsigned int i=0; i<current_point.size(); i++){
    current_point[i] = gradient_terms[i]/gradient_norm;
    //cerr<<"The current point was "<<current_point[i]<<" and the gradient term was "<<gradient_terms[i]<<endl;
  }
}

void expGradientUpdate(
    vector<vector<float> > &ef_current_points,
    vector<vector<float> > &ef_new_points,
    vector<vector<float> > &fe_current_points,
    vector<vector<float> > &fe_new_points,
    vector<vector<float> > &ef_gradients,
    vector<vector<float> > &fe_gradients,
    float step_size){
  // MAKING UPDATES IN THE E GIVEN F DIRECTION
 
  unsigned int num_conditionals = ef_new_points.size();
  
  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate (num_conditionals)
  for (unsigned int i=2; i<num_conditionals; i++) {
    expGradientUpdateSingleGroup(
        ef_current_points[i],
        ef_new_points[i],
        ef_gradients[i],
        step_size);
  }
  
  num_conditionals = fe_new_points.size();
  
  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate (num_conditionals)
  for (unsigned int i=2; i<num_conditionals; i++) {
    expGradientUpdateSingleGroup(
        fe_current_points[i],
        fe_new_points[i],
        fe_gradients[i],
        step_size);
  }

}

void uniformizeProbs(vector<vector<float> >&points){
  unsigned int num_conditionals = points.size();

  //cerr<<"number of threads is "<<omp_get_num_threads()<<endl;
  #pragma omp parallel for firstprivate(num_conditionals)
  for (unsigned int i=2; i<num_conditionals; i++) {
    unsigned int point_size = points[i].size();
    for (unsigned int j=0; j<point_size; j++) {
      points[i][j] = 1.0/(float)point_size;
    }
  }
  cerr<<"points are "<<points<<endl;
}

float exponentiatedGradient(const vector<vector<float> > & ef_expected_counts,
    const vector<vector<float> > & ef_current_probs,
    const vector<float> & ef_rowwiseExpCntsSum,
    vector<vector<float> > & ef_optimized_probs,
    const vector<vector<float> > & fe_expected_counts,
    const vector<vector<float> > & fe_current_probs,
    const vector<float> & fe_rowwiseExpCntsSum,
    vector<vector<float> > & fe_optimized_probs,
    float lambda_ef,
    float lambda_fe,
    float reg_lambda,
    const vector<vector<pair<unsigned int,unsigned int> > > &ef_map,
    vcbList &eTrainVcbList,
    vcbList &fTrainVcbList,
    regularization_type reg_option,
    regularization_func_type reg_func_option) {
   float eta = ETA;
  cerr<<"the regularization option is "<<reg_option<<endl;
	
  //cout<<"projected gradient descent here"<<endl;
  //COPYING THE CURRENT POINT
  vector<vector<float> > ef_current_points(ef_current_probs);
  vector<vector<float> > fe_current_points(fe_current_probs);
  // AS INITIALIZATION, JUST UNIFORMINZE THE PROBABILITIES. 
  // DONT KNOW IF THIS WILL WORK
  //uniformizeProbs(ef_current_points);
  //uniformizeProbs(fe_current_points);

  //cout<<"the size of ef current points is "<<ef_current_points<<endl;
  //cout<<"the size of fe current points is "<<fe_current_points<<endl;
	//cout <<"the number of PGD iterations is "<<NUM_PGD_ITERATIONS<<endl;
  float current_function_value;
  float exp_complete_ll_term_value = 0.;
  float reg_term_value=0;
  //For the null word, i.e. the zeroth word, we should just run the regular
  //m-step in each direction.
  reNormalize(
    ef_expected_counts[0],
    ef_rowwiseExpCntsSum[0],
    ef_current_points[0]);
  reNormalize(
    fe_expected_counts[0],
    fe_rowwiseExpCntsSum[0],
    fe_current_points[0]);
  cerr<<"Finished renormalizing the null probs"<<endl;
  //Running PGD for the rest of the parameters
  evalFunction(
    ef_expected_counts,
    ef_current_points,
    ef_rowwiseExpCntsSum,
    fe_expected_counts,
    fe_current_points,
    fe_rowwiseExpCntsSum,
    lambda_ef,
    lambda_fe,
    reg_lambda,
    ef_map,
    eTrainVcbList,
    fTrainVcbList,
    reg_option,
    reg_func_option,
    &exp_complete_ll_term_value,
    &reg_term_value);
  current_function_value = exp_complete_ll_term_value + reg_term_value;
  float current_reg_term_value = reg_term_value;
  cerr<<"The starting expected complete data log likelihood was "<<exp_complete_ll_term_value<<endl;
  cerr<<"The starting reg term was "<<reg_term_value<<endl;
  cerr<<"The starting function value is: "<<current_function_value<<endl;

  cerr<<"Performing "<<NUM_PGD_ITERATIONS<< "exponentiated gradient iterations"<<endl;
  float best_func_value = current_function_value;
  float best_reg_term_value = reg_term_value;

  cerr<<"The best func value before staring line search was "<<best_func_value<<endl;
  cerr<<"The current function value before staring line search was "<<current_function_value<<endl;
	for (int time = 1; time <= NUM_PGD_ITERATIONS; time++)
	{
		cerr<<"time: "<<time<<" eta:"<<eta<<endl;

    //cerr<<"initializing current gradient"<<endl;
    // INITIAZLIZING THE CURRENT GRADIENTS 
		vector<vector<float> > ef_gradients,fe_gradients;
    ef_gradients = vector<vector<float> >(ef_current_probs.size());
    fe_gradients = vector<vector<float> >(fe_current_probs.size());
    initGradientsToZero(
        ef_gradients,
        ef_current_points,
        fe_gradients,
        fe_current_points);

    //cerr<<"evaluating current gradient"<<endl;
		evalGradient(
      ef_expected_counts,
      ef_current_points,
      ef_rowwiseExpCntsSum,
      ef_gradients,
      fe_expected_counts,
      fe_current_points,
      fe_rowwiseExpCntsSum,
      fe_gradients,
      lambda_ef,
      lambda_fe,
      reg_lambda,
      ef_map,
      eTrainVcbList,
      fTrainVcbList,
      reg_option,
      reg_func_option);
    
    vector<vector<float> > ef_new_points,fe_new_points;
    ef_new_points = vector<vector<float> >(ef_current_probs.size());
    fe_new_points = vector<vector<float> >(fe_current_probs.size());
    zeroInitVectorOfVector(ef_current_points,
        ef_new_points);
    zeroInitVectorOfVector(fe_current_points,
        fe_new_points);
		
		float final_alpha = 0.0 ; //if the function value does not improve at all, then the armijo beta should be 1

    float func_value_at_temp_point;
    float exp_complete_ll_term_at_temp_point_value = 0;
    float reg_term_at_temp_point_value =0;
    
    expGradientUpdate(
        ef_current_points,
        ef_new_points,
        fe_current_points,
        fe_new_points,
        ef_gradients,
        fe_gradients,
        eta);

    //TAKE EXPONENTIALTED GRAIDENT STEP 
    //
    evalFunction(
      ef_expected_counts,
      ef_current_points,
      ef_rowwiseExpCntsSum,
      fe_expected_counts,
      fe_current_points,
      fe_rowwiseExpCntsSum,
      lambda_ef,
      lambda_fe,
      reg_lambda,
      ef_map,
      eTrainVcbList,
      fTrainVcbList,
      reg_option,
      reg_func_option,
      &exp_complete_ll_term_at_temp_point_value,
      &reg_term_at_temp_point_value);
    func_value_at_temp_point = exp_complete_ll_term_at_temp_point_value + reg_term_at_temp_point_value;
    cerr<<"The Function value after exp gradient update is "<<func_value_at_temp_point<<endl;
    // IF THE FUNCTION VALUE DID NOT CHANGE, THEN BREAK
    // IF THE FUCTION VALUE REDUCED, THEN HALVE THE STEP SIZE
    // IF THE FUNCTION VALUE IMPROVED, THEN INCREASE THE STEP SIZE
    // SLIGHTLY
    if (current_function_value == func_value_at_temp_point) {
      cerr<<"Terminating early in iteration "<<time<<" since there was no "<<
        " change in the function value "<<endl;
      break;
    } else if (func_value_at_temp_point < current_function_value) {
      //cerr<<"The function value at temp point was better since the current function value was "<<current_function_value<<endl;
      eta *= 2;  
    } else {
      //cerr<<"The function value at temp point was worse"<<endl;
      eta *= 0.5;
    }
    if (func_value_at_temp_point < best_func_value) {
      best_func_value = func_value_at_temp_point;
    }

    current_function_value = func_value_at_temp_point;

	}
  // IT MAKES SENSE TO SET THE CURRENT LEARNING RATE TO the most recent one
  //ETA=eta;
  cerr<<"After exponentiated gradient, the func value at temp point was "<<best_func_value<<endl;
  cerr<<"After exponentiated gradient, the new step size was "<<ETA<<endl;
	//new_prob = current_point;
  //Storing the optimized probs in the new point
  ef_optimized_probs = ef_current_points;

  fe_optimized_probs = fe_current_points;
  return(current_reg_term_value);
 
}

float projectedGradientDescentWithArmijoRule(
    const vector<vector<float> > & ef_expected_counts,
    const vector<vector<float> > & ef_current_probs,
    const vector<float> & ef_rowwiseExpCntsSum,
    vector<vector<float> > & ef_optimized_probs,
    const vector<vector<float> > & fe_expected_counts,
    const vector<vector<float> > & fe_current_probs,
    const vector<float> & fe_rowwiseExpCntsSum,
    vector<vector<float> > & fe_optimized_probs,
    float lambda_ef,
    float lambda_fe,
    float reg_lambda,
    const vector<vector<pair<unsigned int,unsigned int> > > &ef_map,
    vcbList &eTrainVcbList,
    vcbList &fTrainVcbList,
    regularization_type reg_option,
    regularization_func_type reg_func_option) {
  float eta = ETA;
  cerr<<"the regularization option is "<<reg_option<<endl;
  /*
  cout<<"The ef expected counts are "<<ef_expected_counts<<endl;
  cout<<"The fe expected counts are "<<fe_expected_counts<<endl;
  getchar();
  cout<<"The ef current point is "<<ef_current_probs<<endl;
  cout<<"The fe current point is "<<fe_current_probs<<endl;
  getchar();
  */
	//cout<<"projected gradient descent here"<<endl;
  //COPYING THE CURRENT POINT
  vector<vector<float> > ef_current_points(ef_current_probs);
  vector<vector<float> > fe_current_points(fe_current_probs);
  //cout<<"the size of ef current points is "<<ef_current_points<<endl;
  //cout<<"the size of fe current points is "<<fe_current_points<<endl;
	//cout <<"the number of PGD iterations is "<<NUM_PGD_ITERATIONS<<endl;
  float current_function_value;
  float exp_complete_ll_term_value = 0.;
  float reg_term_value=0;
  //For the null word, i.e. the zeroth word, we should just run the regular
  //m-step in each direction.
  reNormalize(
    ef_expected_counts[0],
    ef_rowwiseExpCntsSum[0],
    ef_current_points[0]);
  reNormalize(
    fe_expected_counts[0],
    fe_rowwiseExpCntsSum[0],
    fe_current_points[0]);
  cerr<<"Finished renormalizing the null probs"<<endl;
  //Running PGD for the rest of the parameters
  evalFunction(
    ef_expected_counts,
    ef_current_points,
    ef_rowwiseExpCntsSum,
    fe_expected_counts,
    fe_current_points,
    fe_rowwiseExpCntsSum,
    lambda_ef,
    lambda_fe,
    reg_lambda,
    ef_map,
    eTrainVcbList,
    fTrainVcbList,
    reg_option,
    reg_func_option,
    &exp_complete_ll_term_value,
    &reg_term_value);
  current_function_value = exp_complete_ll_term_value + reg_term_value;
  float current_reg_term_value = reg_term_value;
  cerr<<"The starting expected complete data log likelihood was "<<exp_complete_ll_term_value<<endl;
  cerr<<"The starting reg term was "<<reg_term_value<<endl;
  cerr<<"The starting function value is: "<<current_function_value<<endl;

  cerr<<"Performing "<<NUM_PGD_ITERATIONS<< "PGD iterations"<<endl;
	for (int time = 1; time <= NUM_PGD_ITERATIONS; time++)
	{
    //eta = eta/time;
		cerr<<"time: "<<time<<" eta:"<<eta<<endl;

    //cerr<<"initializing current gradient"<<endl;
    // INITIAZLIZING THE CURRENT GRADIENTS 
		vector<vector<float> > ef_gradients,fe_gradients;
    ef_gradients = vector<vector<float> >(ef_current_probs.size());
    fe_gradients = vector<vector<float> >(fe_current_probs.size());
    initGradientsToZero(
        ef_gradients,
        ef_current_points,
        fe_gradients,
        fe_current_points);
    //cerr<<"evaluating current gradient"<<endl;
		evalGradient(
      ef_expected_counts,
      ef_current_points,
      ef_rowwiseExpCntsSum,
      ef_gradients,
      fe_expected_counts,
      fe_current_points,
      fe_rowwiseExpCntsSum,
      fe_gradients,
      lambda_ef,
      lambda_fe,
      reg_lambda,
      ef_map,
      eTrainVcbList,
      fTrainVcbList,
      reg_option,
      reg_func_option);
    
    vector<vector<float> > ef_new_points,fe_new_points;
    ef_new_points = vector<vector<float> >(ef_current_probs.size());
    fe_new_points = vector<vector<float> >(fe_current_probs.size());
    zeroInitVectorOfVector(ef_current_points,
        ef_new_points);
    zeroInitVectorOfVector(fe_current_points,
        fe_new_points);
    //cout<<"The size of ef_new points is "<<ef_new_points<<endl;
    //cout<<"The size of fe_new points is "<<fe_new_points<<endl;
    //cout<<"ef gradients is "<<ef_gradients<<endl;
    //cout<<"fe gradients is "<<fe_gradients<<endl;
    /*
    initGradientsToZero(
        ef_gradients,
        ef_new_points,
        fe_gradients,
        fe_new_points);
    */
		//moving in the opposite direction of the gradient
    //
    takeGradientStep(ef_new_points,
        ef_current_points,
        ef_gradients,
        fe_new_points,
        fe_current_points,
        fe_gradients,
        eta);
    //cout<<"We just took a gradient step"<<endl;
		vector<vector<float> > ef_new_feasible_points,fe_new_feasible_points;
    ef_new_feasible_points = vector<vector<float> >(ef_new_points.size());
    fe_new_feasible_points = vector<vector<float> >(fe_new_points.size());

    zeroInitVectorOfVector(ef_current_points,
        ef_new_feasible_points);
    zeroInitVectorOfVector(fe_current_points,
        fe_new_feasible_points);
    /*
    initGradientsToZero(
        ef_gradients,
        ef_new_feasible_points,
        fe_gradients,
        fe_new_feasible_points);
    */
    //cout<<"Projecting points onto the simplex"<<endl;
		// PROJECTING THE POINTS ON THE SIMPLEX
		projectPointsOntoSimplex(ef_new_points,ef_new_feasible_points);
		projectPointsOntoSimplex(fe_new_points,fe_new_feasible_points);
		//printVector(new_feasible_point);
		//cout<<"feasible point dimension is "<<new_feasible_point.size()<<endl;
		//int num_zero_entries = 0;
		
		//for (int i = 0;i<new_feasible_point.size();i++)
		//{
		//	if (new_feasible_point[i] == 0)
		//	{
		//		num_zero_entries++ ;
		//	}
		//}
    
		//cout<<"the number of zero entries is "<<num_zero_entries<<endl;
		//getchar();
		//cout<<"armijo beta is "<<ARMIJO_BETA<<endl;
		//cout<<"armijo sigma is "<<ARMIJO_SIGMA<<endl;
    /*
    //COMPUTING THE ARMIJO BOUND. FOR NOW, THERE IS NO NEED TO COMPUTE IT. JUST DO LINE SEARCH. 
		float armijo_bound = 0.0;
		for (int i=0 ;i<num_elements;i++)
		{
			float bound_term = ARMIJO_SIGMA * ARMIJO_BETA * gradient[i] * (new_feasible_point[i] - current_point[i]); 	
			//cout<<"the grad is "<<gradient[i]<<" the new feasible point is "<<new_feasible_point[i]<<" current point is "<<current_point[i]<<endl;
			//cout<<"the bound term is "<<bound_term<<endl;
			armijo_bound -= bound_term;
			//cout<<"temp armijo bound "<<armijo_bound<<endl;
		}
		*/
		//getchar();
    bool armijo_bound = 0;
		bool terminate_line_srch = 0;
		int num_steps = 1;
		float current_alpha = ARMIJO_BETA ;
		float final_alpha = 0.0 ; //if the function value does not improve at all, then the armijo beta should be 1
		float current_armijo_bound = armijo_bound;
		float best_func_value = current_function_value;
    float best_reg_term_value = reg_term_value;
		bool no_update = 1;
		//cout<<"current function value is "<<current_function_value<<endl;
		//printf ("current function value is %.15f\n",current_function_value);
    cerr<<"The best func value before staring line search was "<<best_func_value<<endl;
		while(terminate_line_srch != 1 && num_steps <= 20)
		{	
			//cout<<"current armijo bound is "<<current_armijo_bound<<endl;
			//cout<<"we are in teh while loop"<<endl;
			//cout<<"num steps is "<<num_steps<<endl;
		//	current_beta = 
      vector<vector<float> > ef_temp_points,fe_temp_points;
      ef_temp_points = vector<vector<float> >(ef_new_points.size());
      fe_temp_points = vector<vector<float> >(fe_new_points.size());

      zeroInitVectorOfVector(ef_current_points,
          ef_temp_points);
      zeroInitVectorOfVector(fe_current_points,
          fe_temp_points);

      /*
      initGradientsToZero(
          ef_gradients,
          ef_temp_points,
          fe_gradients,
          fe_temp_points);
      */

      getInterpolatedPoints(ef_new_feasible_points,
        ef_current_points,
        ef_temp_points,
        fe_new_feasible_points,
        fe_current_points,
        fe_temp_points,
        current_alpha);
      //cout<<"After interpolation, the ef point is "<<ef_temp_points<<endl;
      //cout<<"After interpolation, the fe point is "<<fe_temp_points<<endl;

      /*
			for (int i =0 ;i<num_elements;i++)		
			{
				temp_point[i] = (1.0 - current_alpha) * current_point[i] + current_alpha * new_feasible_point[i];
			}
      */
      float func_value_at_temp_point;
      float exp_complete_ll_term_at_temp_point_value = 0;
      float reg_term_at_temp_point_value =0;
      evalFunction(
        ef_expected_counts,
        ef_temp_points,
        ef_rowwiseExpCntsSum,
        fe_expected_counts,
        fe_temp_points,
        fe_rowwiseExpCntsSum,
        lambda_ef,
        lambda_fe,
        reg_lambda,
        ef_map,
        eTrainVcbList,
        fTrainVcbList,
        reg_option,
        reg_func_option,
        &exp_complete_ll_term_at_temp_point_value,
        &reg_term_at_temp_point_value);
      func_value_at_temp_point = exp_complete_ll_term_at_temp_point_value + reg_term_at_temp_point_value;
			//cerr<<"function value at alpha"<<current_alpha<<" is "<<func_value_at_temp_point<<endl;
			//printf ("function value at temp point is %.15f and the iteration number is %d \n",func_value_at_temp_point,num_steps);
			//printf ("current alpha is %.15f\n",current_alpha);
			//getchar();
			if (func_value_at_temp_point < best_func_value)
			{
				best_func_value = func_value_at_temp_point;
        best_reg_term_value = reg_term_at_temp_point_value;
				final_alpha = current_alpha;
				no_update = 0;
				//cout<<"we arrived at a better function value"<<endl;
				//getchar();
			}
			/*
			if (current_function_value - func_value_at_temp_point >= current_armijo_bound)
			{
				//cout<<"the terminate line src condition was met "<<endl;
				terminate_line_srch = 1;
			}
      */
			current_alpha *= ARMIJO_BETA;
			current_armijo_bound *= ARMIJO_BETA;
			num_steps += 1;
			//getchar();
		}
    //cerr<<"The best function value at the end of line search was "<<best_func_value<<" at alpha "<<final_alpha<<endl;
		//printf ("final alpha was %f\n",final_alpha);
		//cout<<"the value of not update was "<<no_update<<endl;
		//getchar();
	
		//vector<float> next_point ;
		if (no_update == 0)
		{
      //cout<<"Best func value was "<<best_func_value<<endl;
      getInterpolatedPoints(ef_new_feasible_points,
        ef_current_points,
        ef_current_points,
        fe_new_feasible_points,
        fe_current_points,
        fe_current_points,
        final_alpha);
      current_function_value = best_func_value;
      current_reg_term_value = best_reg_term_value;
      /*
			//next_point.resize(num_elements);
			for (int i =0 ;i<num_elements;i++)
			{
				float coordinate_point = (1.0 - final_alpha)*current_point[i] + final_alpha * new_feasible_point[i];
				//next_point.push_back(coordinate_point);
				current_point[i] = coordinate_point;
			}
			//current_point = next_point;
      */
		}
		else
		{
			cerr<<"Terminating early from PGD because there was no update in line search"<<endl;
			break;
		}

	}		
	//new_prob = current_point;
  //Storing the optimized probs in the new point
  ef_optimized_probs = ef_current_points;
  /*
  cout<<"The size of ef_optimized probs is "<<ef_optimized_probs.size()<<endl;
  for (int i=0;i<ef_optimized_probs.size();i++) {
    cout<<"ith vector is "<<endl;
    printVector(ef_optimized_probs[i]);
  }
  getchar();
  */
  fe_optimized_probs = fe_current_points;
  return(current_reg_term_value);
}



