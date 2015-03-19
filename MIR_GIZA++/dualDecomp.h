#ifndef DUAL_DECOMP_
#define DUAL_DECOMP_

#include <vector>
#include "model1.h"
#include "vocab.h"
//#include "projectedGradientDescent.h"
#include "projectedGradientDescentDualDecomp.h"
#include <omp.h>

class dualDecomp {
  public: 
     //vector<vector<pair<unsigned int,unsigned int> > > ef_map;
 /*
  dualDecomp():
    ef_map(vector<vector<pair<unsigned int,unsigned int> > > ()) {
  }
  
  template<class alignment_model> 
  void buildEFMap(alignment_model &ef_model,alignment_model &fe_model) {
    cerr<<"We are building an ef map inside dual decomp"<<endl;
    ef_model.getTtable().buildEFMap(ef_map,fe_model.getTtable().getLexmat());
  }
  */

  //template<class alignment_model>
  //void runDDMStep(alignment_model &ef_model, alignment_model &fe_model) {
  //}
  /*
  template<class alignment_model>
  void runDDMStep(
      alignment_model &ef_model,
      alignment_model &fe_model,
      const float reg_lambda,
      vcbList &eTrainVcbList,
      vcbList &fTrainVcbList);

  void reNormalize(
    const vector<float> &expected_counts,
    const float expected_counts_sum,
    vector<float> &new_point);
  */
  /*
  static void zeroInitVectorOfVector(
      const vector<vector<float> > &source,
      vector<vector<float> > &target) {
    int num_conditionals = source.size();
    //#pragma omp parallel for firstprivate(num_conditionals)
    for (int i=0; i<num_conditionals; i++) {
      int target_row_size = source[i].size();
      target[i] = vector<float> (target_row_size,0.0);
    }
  }
  */

  template<class alignment_model>
  static float runDDMStep(
      alignment_model &ef_model,
      alignment_model &fe_model,
      const float reg_lambda,
      vcbList &eTrainVcbList,
      vcbList &fTrainVcbList,
       vector<vector<pair<unsigned int,unsigned int> > > &ef_map) {
    float reg_term_value = 0;
    cerr<<"Running the DD m-step"<<endl;
    //running the PGD m step
    // STORING EXPECTED COUNTS
    vector<vector<float> > ef_expCntsVec,ef_probsVec;
    vector<vector<float> > fe_expCntsVec,fe_probsVec;
    vector<vector<float> > ef_optimizedProbs,fe_optimizedProbs;
    vector<float> ef_rowwiseExpCntsSum,fe_rowwiseExpCntsSum;
    cerr<<" ACCUMULATING EXPECTED COUNTS FROM E given F"<<endl;
    ef_model.getTtable().getCounts(&ef_expCntsVec,&ef_rowwiseExpCntsSum);
    //printCounts(ef_expCntsVec);
    //getchar();
    cerr<<" ACCUMULATING PROBABILITIES FROM E GIVEN F"<<endl;
    ef_model.getTtable().getProbs(&ef_probsVec);
    //cerr<<"Printing the ef probs"<<endl;
    //printCounts(ef_probsVec);
    //getchar();
    cerr<<" ACCUMULATING EXPECTED COUNTS FROM F given E"<<endl;
    fe_model.getTtable().getCounts(&fe_expCntsVec,&fe_rowwiseExpCntsSum);
    //printCounts(ef_expCntsVec);
    //getchar();

    cerr<<" ACCUMULATING PROBABILITIES FROM F GIVEN E"<<endl;
    fe_model.getTtable().getProbs(&fe_probsVec);
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
   
    //cerr<<"Initialializing lagrange multipliers and points to zero"<<endl;
    //cerr<<"expected counts are "<<ef_expCntsVec<<endl;
    //Initialize the lagrange multipliers with zero
    vector<vector<float> > current_lagrange_multipliers(ef_expCntsVec.size());
    zeroInitVectorOfVector(ef_expCntsVec,
        current_lagrange_multipliers);
    
    //cerr<<"Initialized lagrange multipliers and points to zero"<<endl;
    // Since the lagrange multipliers have been initialized to zero
    // the first step is just like the standard m step i.e. count 
    // and divide.
    vector<vector<float> > ef_current_point(ef_expCntsVec.size());
    zeroInitVectorOfVector(ef_expCntsVec,
        ef_current_point);

    vector<vector<float> > fe_current_point(fe_expCntsVec.size());
    zeroInitVectorOfVector(fe_expCntsVec,
        fe_current_point);
    
    //cerr<<"Generating the ef starting point "<<endl;
    //Normalizing expected counts
    for(unsigned int i=0; i<ef_expCntsVec.size(); i++){
      if (ef_expCntsVec[i].size()>=1) {
        //cerr<<"The rowwise expected counts sum  was "<<ef_rowwiseExpCntsSum[i]<<endl;
        reNormalize(
            ef_expCntsVec[i],
            ef_rowwiseExpCntsSum[i],
            ef_current_point[i]);
      }
    }
    //cerr<<"After re normalizing, the ef points are "<<ef_current_point<<endl;
    //cerr<<"Generating the fe starting point "<<endl;
    //Normalizing expected counts
    for(unsigned int i=0; i<fe_expCntsVec.size(); i++){
      if (fe_expCntsVec[i].size()>=1) {
        reNormalize(
            fe_expCntsVec[i],
            fe_rowwiseExpCntsSum[i],
            fe_current_point[i]);
      }
    }
    //cerr<<"After re normalizing, the fe points are "<<fe_current_point<<endl;
    vector<vector<float> > current_ef_lagrange_multipliers(ef_expCntsVec.size());
    zeroInitVectorOfVector(ef_expCntsVec,
        current_ef_lagrange_multipliers);

    vector<vector<float> > current_fe_lagrange_multipliers(fe_expCntsVec.size());
    zeroInitVectorOfVector(fe_expCntsVec,
        current_fe_lagrange_multipliers);

    //Since we are done normalizing, we will now start DD
    //first update the lagrange multipliers
    
    for (int it=0; it<10; it++) {
      reg_term_value = 0.;
      int num_conditionals = ef_map.size();
      for (int i =2 ;i<num_conditionals;i++)
     { 
        if (ef_rowwiseExpCntsSum[i] == 0.) {
          continue;
        }

          //UPDATING THE LAGRANGE MULTIPLIERS
          for (unsigned int j=0; j<ef_map[i].size(); j++) {
            //cerr<<"the item is "<<j<<endl;
            unsigned int f_position =  ef_map[i][j].first;
            unsigned int e_position_in_f_row = ef_map[i][j].second;
            float diff = reg_lambda*(ef_current_point[i][j]*fTrainVcbList.getProbForWord(i) -
                fe_current_point[f_position][e_position_in_f_row]*eTrainVcbList.getProbForWord(f_position));
            current_lagrange_multipliers[i][j] -= diff;
            reg_term_value += diff;
            current_ef_lagrange_multipliers[i][j] = current_lagrange_multipliers[i][j];
            current_fe_lagrange_multipliers[f_position][e_position_in_f_row] = current_lagrange_multipliers[i][j];
          }
      }
      cerr<<"The reg term value in iteration "<<it<<" of DD was "<<reg_term_value<<endl;
      vector<vector<float> > ef_new_point(ef_expCntsVec.size());
      zeroInitVectorOfVector(ef_expCntsVec,
          ef_new_point);

      vector<vector<float> > fe_new_point(fe_expCntsVec.size());
      zeroInitVectorOfVector(fe_expCntsVec,
          fe_new_point);
      //cerr<<"The ef expected counts are "<<ef_expCntsVec<<endl;
      //cerr<<"The fe expected counts are "<<fe_expCntsVec<<endl;
      // Now to run PGD on the decoupled problems. First the ef problems
      cerr<<"Running PGD on the decoupled ef problems"<<endl;
      num_conditionals = ef_expCntsVec.size();
      //#pragma omp parallel for firstprivate(num_conditionals)
      for (int i =2 ;i<num_conditionals;i++)
      { 
        if (ef_rowwiseExpCntsSum[i] == 0.) {
          continue;
        }
        //cerr<<"Running pgd"<<endl;
        //cerr<<"sending "<<i<<" ef point "<<ef_current_point[i]<<endl;
        projectedGradientDescentWithArmijoRuleDualDecomp(ef_expCntsVec[i],
          ef_current_point[i],
          ef_new_point[i],
          current_ef_lagrange_multipliers[i],
          1,
          fTrainVcbList.getProbForWord(i),
          reg_lambda);
        //cerr<<"Finished pgd"<<endl;
      }

      ef_current_point = ef_new_point;

      cerr<<"Running PGD on the decoupled fe problems"<<endl;
      num_conditionals = fe_expCntsVec.size();
      //#pragma omp parallel for firstprivate(num_conditionals)
      for (int i =2 ;i<num_conditionals;i++)
      { 
        if (fe_rowwiseExpCntsSum[i] == 0.) {
          continue;
        }
        //cerr<<"sending "<<i<<" fe point "<<fe_current_point[i]<<endl;
        projectedGradientDescentWithArmijoRuleDualDecomp(fe_expCntsVec[i],
          fe_current_point[i],
          fe_new_point[i],
          current_fe_lagrange_multipliers[i],
          -1,
          eTrainVcbList.getProbForWord(i),
          reg_lambda);
      }
      fe_current_point = fe_new_point;
    }
    /*
    float reg_term_value = projectedGradientDescentWithArmijoRule(
      ef_expCntsVec,
      ef_probsVec,
      ef_rowwiseExpCntsSum,
      ef_optimizedProbs,
      fe_expCntsVec,
      fe_probsVec,
      fe_rowwiseExpCntsSum,
      fe_optimizedProbs,
      reg_lambda,
      ef_map,
      eTrainVcbList,
      fTrainVcbList,
      reg_option,
      reg_func_option);
    */
    // After running PGD, we need to assign probs into the
    // t-table
    ef_model.getMutableTtable().setProbs(ef_current_point);
    fe_model.getMutableTtable().setProbs(fe_current_point);
    //return(reg_term_value);
    return(reg_term_value);
  }


  static void reNormalize(
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
      //cerr<<"The size of expected counts was "<<expected_counts.size()<<endl;
      //cerr<<"The size of new point was "<<new_point.size()<<endl;
      for (unsigned int i=0;i<new_point.size();i++) {
        //cerr<<"expected count was "<<expected_counts[i]<<endl;
        //cerr<<"sum was "<<expected_counts_sum<<endl;
        new_point[i] = expected_counts[i]/expected_counts_sum;
      }
    }
    return;
  }

};
#endif
