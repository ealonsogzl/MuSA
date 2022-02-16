
real = read.table('/home/alonsoe/GIT/thermal_DA/Outputs/hightatras_original.csv', header = T)

OL = read.csv('/home/alonsoe/GIT/thermal_DA/Outputs/MuSA_out/OL_0_3.csv')
DA0 = read.csv('/home/alonsoe/GIT/thermal_DA/Outputs/MuSA_out/updated_0_3.csv')
DA3 = read.csv('/home/alonsoe/GIT/thermal_DA/Outputs/MuSA_out/updated_3_3.csv')
DA5 = read.csv('/home/alonsoe/GIT/thermal_DA/Outputs/MuSA_out/updated_5_3.csv')
DA10 = read.csv('/home/alonsoe/GIT/thermal_DA/Outputs/MuSA_out/updated_10_3.csv')


par(mfrow = c(4,1))
plot(real$snw[15000:48000],t='l', ylim = c(0,110), main= 'Frequency: 1 day', ylab = 'SWE [mm]')
lines(OL$X2[15000:48000], col = 'red')
lines(DA0$X2[15000:48000], col = 'blue')
legend('topright', legend=c("'real'", "OL", "updated"),
       col=c("black", "red", 'blue'), lty=1, cex=0.8)


plot(real$snw[15000:48000],t='l', ylim = c(0,110), main= 'Frequency: 4 day', ylab = 'SWE [mm]')
lines(OL$X2[15000:48000], col = 'red')
lines(DA3$X2[15000:48000], col = 'blue')
legend('topright', legend=c("'real'", "OL", "updated"),
       col=c("black", "red", 'blue'), lty=1, cex=0.8)

plot(real$snw[15000:48000],t='l', ylim = c(0,110), main= 'Frequency: 6 day', ylab = 'SWE [mm]')
lines(OL$X2[15000:48000], col = 'red')
lines(DA5$X2[15000:48000], col = 'blue')
legend('topright', legend=c("'real'", "OL", "updated"),
       col=c("black", "red", 'blue'), lty=1, cex=0.8)

plot(real$snw[15000:48000],t='l', ylim = c(0,110), main= 'Frequency: 11 day', ylab = 'SWE [mm]')
lines(OL$X2[15000:48000], col = 'red')
lines(DA10$X2[15000:48000], col = 'blue')
legend('topright', legend=c("'real'", "OL", "updated"),
       col=c("black", "red", 'blue'), lty=1, cex=0.8)

