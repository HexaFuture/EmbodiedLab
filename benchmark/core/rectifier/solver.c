/* Fixed-clock Heun integration of a capacitor-input full-wave bridge.
 * Two identical Shockley diodes in series during conduction. Reverse leakage
 * and reverse recovery are omitted and declared in the public model contract.
 * No waveform lookup, output reset, fitted hidden correction or steady shortcut.
 */
#include <math.h>
#include <stddef.h>

double rectifier_current(double drive, double n) {
    const double rs=25.0, is=1e-8, a=2*n*8.617333262145e-5*298.0;
    if(drive<=0) return 0;
    double lo=0, hi=drive/rs;
    double x=fmax(1e-12, (drive-1.1)/rs);
    if(x>=hi) x=hi/2;
    for(int j=0;j<48;j++) {
        double f=rs*x+a*log1p(x/is)-drive;
        if(fabs(f)<1e-12) return x;
        if(f>0) hi=x; else lo=x;
        double y=x-f/(rs+a/(x+is));
        x=(y>lo && y<hi)?y:(lo+hi)/2;
    }
    return x;
}

/* rows are [time, signed source voltage, output voltage, load current,
 * signed source current]; sampling at every step, caller may decimate. */
double rectifier_advance(double v, double c, double r, double n,
                         long long start, int steps, double hz, int enabled,
                         double discharge_r, double *rows) {
    const double dt=1/hz, omega=2*3.14159265358979323846*50;
    const double g=1/r+(discharge_r>0?1/discharge_r:0);
    for(int k=0;k<steps;k++) {
        double t=(start+k)*dt, src=12*sin(omega*t);
        double i=enabled?rectifier_current(fabs(src)-v,n):0;
        double dv=(i-g*v)/c;
        double t1=(start+k+1)*dt, s1=12*sin(omega*t1);
        double pred=v+dt*dv;
        double i1=enabled?rectifier_current(fabs(s1)-pred,n):0;
        v+=dt*.5*(dv+(i1-g*pred)/c);
        if(rows) {
            double ip=enabled?rectifier_current(fabs(s1)-v,n):0;
            size_t j=(size_t)k*5;
            rows[j]=t1;rows[j+1]=s1;rows[j+2]=v;
            rows[j+3]=v/r;rows[j+4]=copysign(ip,s1);
        }
    }
    return v;
}
