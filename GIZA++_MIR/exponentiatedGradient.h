using namespace std;
#include <math.h>


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
      <<"not equal to the gradient size"<<endl;
  }
  vector<float> gradient_terms(gradient.size(),0.0);
  // FIRST COMPUTE THE GRADIENT NORMALIZER
  for (unsigned int i=0; i<current_point.size(); i++){
    gradient_terms[i] = current_point[i]*exp(-step_size*gradient[i]);
    gradient_norm += gradient_terms[i];
  }

  for (unsigned int i=0; i<current_point.size(); i++){
    current_point[i] = gradient_terms[i]/gradient_norm;
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
  int num_conditionals = ef_new_points.size();
  #pragma omp parallel for(num_conditionals)
  for (unsigned int i=2; i<num_conditionals; i++) {
    expGradientUpdateSingleGroup(
        ef_current_points[i],
        ef_new_points[i],
        ef_gradients[i],
        step_size);
  }
  num_conditionals = fe_new_points.size();
  #pragma omp parallel for(num_conditionals)
  for (unsigned int i=2; i<num_conditionals; i++) {
    expGradientUpdateSingleGroup(
        fe_current_points[i],
        fe_new_points[i],
        fe_gradients[i],
        step_size);
  }

}

float exponentiatedGradient(const vector<vector<float> > & ef_expected_counts,
    const vector<vector<float> > & ef_current_probs,
    const vector<float> & ef_rowwiseExpCntsSum,
    vector<vector<float> > & ef_optimized_probs,
    const vector<vector<float> > & fe_expected_counts,
    const vector<vector<float> > & fe_current_probs,
    const vector<float> & fe_rowwiseExpCntsSum,
    vector<vector<float> > & fe_optimized_probs,
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
		float current_armijo_bound = armijo_bound;
		float best_func_value = current_function_value;
    float best_reg_term_value = reg_term_value;

    cerr<<"The best func value before staring line search was "<<best_func_value<<endl;
    float func_value_at_temp_point;
    float exp_complete_ll_term_at_temp_point_value = 0;
    float reg_term_at_temp_point_value =0;
    
    expGradientUpdate(
        ef_current_points,
        ef_new_points,
        fe_current_points,
        fe_new_points,
        ef_gradients,
        fe_gradient,
        eta);

    //TAKE EXPONENTIALTED GRAIDENT STEP 
    //
    evalFunction(
      ef_expected_counts,
      ef_current_points,
      ef_rowwiseExpCntsSum,
      fe_expected_counts,
      fe_points,
      fe_rowwiseExpCntsSum,
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

	}		
	//new_prob = current_point;
  //Storing the optimized probs in the new point
  ef_optimized_probs = ef_current_points;

  fe_optimized_probs = fe_current_points;
  return(current_reg_term_value);
 
}
