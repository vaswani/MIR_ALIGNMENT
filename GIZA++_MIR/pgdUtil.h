#include <algorithm>
#include <vector>
#include <iostream>
//#include "defs.h"
#include <omp.h>
//#include "vocab.h"
#include <assert.h>

void runModel1Iterations(model1 &m1,
    int Model1_Iterations,
    bool seedModel1,
    Dictionary *dictionary,
    bool useDictionary,
    string model_prefix) {
  cout << "RUNNING MODEL 1"<<endl;
  int minIter=0;
  for (int it=1; it<=Model1_Iterations; it++) {
    minIter=m1.em_with_tricks_e_step(it,
        seedModel1,
        dictionary,
        useDictionary,
        model_prefix);
    //errors=m1.errorsAL();
    /*
    // RUNNING THE MODELS IN BOTH DIRECTOINS
    cout << "RUNNING E GIVEN F DIRECTION"<<endl;
    ef_minIter=ef_m1.em_with_tricks(Model1_Iterations,seedModel1,*dictionary, useDict);
    ef_errors=ef_m1.errorsAL();
    */
  }

}

float runPGDMStep(
    model1 &ef_m1,
    model1 &fe_m1,
    const float reg_lambda_ef,
    const float reg_lambda_fe,
    const vector<vector<pair<unsigned int,unsigned int> > > &ef_map,
    vcbList &eTrainVcbList,
    vcbList &fTrainVcbList) {
  cout<<"Running PGD m-step"<<endl;
  //running the PGD m step
  // STORING EXPECTED COUNTS
  vector<vector<float> > ef_expCntsVec,ef_probsVec;
  vector<vector<float> > fe_expCntsVec,fe_probsVec;
  vector<vector<float> > ef_optimizedProbs,fe_optimizedProbs;
  vector<float> ef_rowwiseExpCntsSum,fe_rowwiseExpCntsSum;
  cout<<" ACCUMULATING EXPECTED COUNTS FROM E given F"<<endl;
  ef_m1.getTtable().getCounts(&ef_expCntsVec,&ef_rowwiseExpCntsSum);
  //printCounts(ef_expCntsVec);
  //getchar();
  cout<<" ACCUMULATING PROBABILITIES FROM E GIVEN F"<<endl;
  ef_m1.getTtable().getProbs(&ef_probsVec);
  //cout<<"Printing the ef probs"<<endl;
  //printCounts(ef_probsVec);
  //getchar();
  cout<<" ACCUMULATING EXPECTED COUNTS FROM E given F"<<endl;
  fe_m1.getTtable().getCounts(&fe_expCntsVec,&fe_rowwiseExpCntsSum);
  //printCounts(ef_expCntsVec);
  //getchar();

  cout<<" ACCUMULATING PROBABILITIES FROM E GIVEN F"<<endl;
  fe_m1.getTtable().getProbs(&fe_probsVec);
  //printCounts(fe_probsVec);
  //getchar();
  //now normalize table
  enum regularization_type reg_option;
  enum regularization_func_type reg_func_option;
  // Contitional reg and joint reg cannot both be 1. 
  // xor gives us that
  if (conditional_reg^joint_reg == 0){
    cerr<<"Warning..."<<endl;
    cerr<<"Ivalid options for regularization"<<endl;
    cerr<<"Setting joint reg to 1"<<endl;
    conditional_reg = 0;
    joint_reg = 1;
  }
  if (l1_reg^l2_reg == 0){
    cerr<<"Warning..."<<endl;
    cerr<<"Ivalid options for regularization type(l1 or l2)"<<endl;
    cerr<<"Setting conditional reg to l2"<<endl;
    l2_reg = 1;
    l1_reg = 0;
  }
  //assert((conditional_reg^joint_reg) != 0 );
  // Same for l2 reg option and l1 reg option
  //assert((l1_reg^l2_reg) != 0);
  if (conditional_reg == 1) {
    reg_option = CONDITIONAL;
  }
  if (joint_reg == 1) {
    reg_option = JOINT;
  }
  if (l1_reg == 1) {
    reg_func_option = L1;
  }
  if (l2_reg == 1) {
    reg_func_option = L2;
  }

  float reg_term_value = projectedGradientDescentWithArmijoRule(
    ef_expCntsVec,
    ef_probsVec,
    ef_rowwiseExpCntsSum,
    ef_optimizedProbs,
    fe_expCntsVec,
    fe_probsVec,
    fe_rowwiseExpCntsSum,
    fe_optimizedProbs,
    reg_lambda_ef,
    reg_lambda_fe,
    ef_map,
    eTrainVcbList,
    fTrainVcbList,
    reg_option,
    reg_func_option);
  // After running PGD, we need to assign probs into the
  // t-table
  /*
  cout<<"printing optimized probs after pgd. The size is"<<ef_optimizedProbs.size()<<endl;
  for (int i=0;i<ef_optimizedProbs.size();i++) {
    cout<<"ith vector is "<<endl;
    printVector(ef_optimizedProbs[i]);
  }
  getchar(); 
  */
  ef_m1.getMutableTtable().setProbs(ef_optimizedProbs);
  fe_m1.getMutableTtable().setProbs(fe_optimizedProbs);
  return(reg_term_value);
}

