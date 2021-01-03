#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Angle normalization convenience function
void wraptopi(double& theta)
{
  while (theta> M_PI) theta-=2.*M_PI;
  while (theta<-M_PI) theta+=2.*M_PI;
}
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);
  // Can be fine-tuned
  P_(3,3) = 0.0025;
  P_(4,4) = 0.0025;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // Can be fine-tuned
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // Can be fine-tuned
  std_yawdd_ = 0.8;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  //Number of tracked states
  n_x_ = 5;    

  //Number of augmented states
  n_aug_ = 7; 

  //Sigma point spreading parameter
  lambda_ = 3 - n_aug_; 

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  //weights_ for calculating mean and covariance from sigma points
  weights_ = VectorXd(2*n_aug_+1); 

  // Computation of weights
  for (int i=0; i< (2*n_aug_)+1; ++i )
  {
      // 0th element exceptional case
      if (i == 0)
        weights_(i) = lambda_/(lambda_+n_aug_);
      else
        weights_(i) = (0.5)/(lambda_+n_aug_);
  }
  
  // Initially set to false, then set to true when process measurement is called for the first time
  is_initialized_ = false;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  
  if (!is_initialized_) //if first measurement
  {
    std::cout << std::endl << "Kalman Filter Initialization " << std::endl;
    // set the state with the initial location and zero velocity
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)//check if lidar measurement
    { // if yes then input the Px and Py value into the state vector
      x_ << meas_package.raw_measurements_[0], // p_x : direct reading from Lidar 
            meas_package.raw_measurements_[1], // p_y : direct reading from Lidar
            0, // Velocity : No Info
            0, // Yaw : No Info
            0; // Yaw Rate : No Info
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)//check if Radar Measurement
    { // if yes then input the Px and Py value into the state vector
      double rho = meas_package.raw_measurements_[0]; // Range
      double phi = meas_package.raw_measurements_[1]; // Bearing
      double rho_dot = meas_package.raw_measurements_[2]; // Range rate
      double p_x = rho * cos(phi); // Polar to Cartesian
      double p_y = rho * sin(phi); // Polar to Cartesian
      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);
      x_ << p_x, // p_x 
            p_y, // p_y 
            v, // Velocity 
            0, // Yaw : No Info
            0; // Yaw Rate : No Info
    }
    
    else 
    {
     std::cout<<"NO MEASUREMENT RECORDED"<< std::endl;// if nothing recorded
    }
    time_us_= meas_package.timestamp_; //input the time value
    is_initialized_ = true; //turn this on as first measurement taken
    
    return;
    
  }
  
  // If not first measurement
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; // time step
  time_us_ = meas_package.timestamp_; // Store time for calculation in next cycle
  
  // Prediction step
  Prediction(dt);//call the prediction step : common for lidar and radar
  
  // Measurement update step
  // If Lidar reading is recieved
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
  UpdateLidar(meas_package);
  }
  // If Radar reading is recieved
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
  UpdateRadar(meas_package);
  }
  
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);
 
  // Fill augmented mean vector
  x_aug.head(5) = x_;
  x_aug(5) = 0.0; // Mean long.acc. noise (Since we are dealing with white noise)
  x_aug(6) = 0.0; // Mean yaw.acc. noise
  
  // fill augmented state covariance
  P_aug.topLeftCorner(5,5) = P_;
  P_aug.block(5,5,2,2) << std_a_*std_a_, 0,
                          0,std_yawdd_*std_yawdd_;
  
  // Cholesky decomposition
  MatrixXd A = MatrixXd(7, 7);
  A = P_aug.llt().matrixL();
  
  // Create augmented sigma points
  double coeff = sqrt(lambda_ + n_aug_);
  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i < n_aug_; ++i)
  {   
      Xsig_aug.col(i+1) = x_aug + coeff*A.col(i); // Positive direction
      Xsig_aug.col(i+1+n_aug_) = x_aug - coeff*A.col(i); // Negative direction
  }

  // Convenience variables for readability 
  VectorXd change = VectorXd(5);
  VectorXd noise = VectorXd(5);

  // Predict sigma points
  for (int i=0; i<2*n_aug_+1; i++)
  {
     // Compute individual state elements for readability
     double v = Xsig_aug.col(i)(2); // Velocity
     double yaw = Xsig_aug.col(i)(3); // Yaw
     double yaw_dot = Xsig_aug.col(i)(4); // Yaw rate
     double nu_a = Xsig_aug.col(i)(5); // Uncertainity in linear acceleration
     double nu_yaw_dot_dot = Xsig_aug.col(i)(6); // Uncertainity in yaw acceleration

     // Check division by zero
     if (yaw_dot > 0.001)
     {
        // Change in state
        change << (v/yaw_dot)*(sin(yaw + yaw_dot*delta_t) - sin(yaw)),
                  (v/yaw_dot)*(-cos(yaw + yaw_dot*delta_t) + cos(yaw)),
                  0,
                  yaw_dot*delta_t,
                  0;
     }
     else
     {
        // Change in state
        change << v*cos(yaw)*delta_t,
                  v*sin(yaw)*delta_t,
                  0,
                  yaw_dot*delta_t,
                  0;
     }

     // Noise vector is common
        noise << (0.5)*(delta_t*delta_t)*cos(yaw)*nu_a,
                (0.5)*(delta_t*delta_t)*sin(yaw)*nu_a,
                delta_t*nu_a,
                (0.5)*(delta_t*delta_t)*nu_yaw_dot_dot,
                delta_t*nu_yaw_dot_dot;

     // Prediction step (Each column of Xsig_pred is filled)
     Xsig_pred_.col(i) = Xsig_aug.col(i).head(5) + change + noise;
  }

  // Computation of predicted mean from sigma points
  x_.fill(0.0);
  for (int i=0; i< (2*n_aug_)+1; ++i )
  {
      // Mean formula 
      x_ += weights_(i)*Xsig_pred_.col(i);
  }
  
  // Predicted covariance 
  P_.fill(0.0);
  for (int i=0; i< (2*n_aug_)+1; ++i  )
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    wraptopi(x_diff(3));
    // Covariance formula
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  
  // Lidar measurement vector dimension
  int n_z = 2;

  // Incoming Lidar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0],   // p_x in m
       meas_package.raw_measurements_[1];   // p_y in m
     
  // Measurement matrix
  Eigen::MatrixXd H = MatrixXd(n_z, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  // Lidar measurement covariance matrix R (2x2)
  Eigen::MatrixXd R = MatrixXd(n_z, n_z);
  R.fill(0.0);
  R(0,0) = std_laspx_*std_laspx_;
  R(1,1) = std_laspy_*std_laspy_;

  // Lidar update steps (Update steps of Linear Kalman Filter)
  // Transform state into measurement space
  VectorXd z_pred = H * x_;

  // Innovation term
  VectorXd y = z - z_pred;

  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;

  // Kalman Gain
  MatrixXd K = PHt * Si;

  // State mean update
  x_ += (K * y);

  // State Covariance update
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // Radar measurement vector dimension  
  int n_z = 3; 

  // Incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0],   // rho in m
       meas_package.raw_measurements_[1],   // phi in rad
       meas_package.raw_measurements_[2];   // rho_dot in m/s
     
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // Calculate measurement sigma points and mean
  Zsig.fill(0.0);
  z_pred.fill(0.0);
  for(int i = 0; i<(2*n_aug_)+1; i++)
  { 
    // Extract state elements for readability
    double p_x = Xsig_pred_.col(i)(0);
    double p_y = Xsig_pred_.col(i)(1);
    double v = Xsig_pred_.col(i)(2);
    double yaw = Xsig_pred_.col(i)(3);
    double yaw_dot = Xsig_pred_.col(i)(4);

    // Transform state elements into measurement space
    double rho = sqrt(p_x*p_x + p_y*p_y);
    double phi = atan2(p_y,p_x);
    double rho_dot = (p_x*cos(yaw)*v + p_y*sin(yaw)*v)/rho;

    // Measurement sigma points 
    Zsig.col(i) << rho,
                   phi,
                   rho_dot;

    // Measurement predicted mean
    z_pred += weights_(i)*Zsig.col(i);

  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    wraptopi(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Radar measurement covariance matrix R (3x3)
  Eigen::MatrixXd R = MatrixXd(n_z, n_z);
  R.fill(0.0);
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;

  S = S + R; // Resulting covariance after gaussian convolution with white noise

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  // Calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    wraptopi(z_diff(1));

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    wraptopi(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc*S.inverse();

  // Innovation
  VectorXd y = z - z_pred;

  // Angle normalization
  wraptopi(y(1));

  // State mean update
  x_ += K*(y);

  // State covariance update
  P_ -= K*S*K.transpose(); 
}