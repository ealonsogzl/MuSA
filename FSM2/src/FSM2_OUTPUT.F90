!-----------------------------------------------------------------------
! Write output
!-----------------------------------------------------------------------
subroutine FSM2_OUTPUT(Npnts,year,month,day,hour,                      &
                       H,LE,LWout,LWsub,Melt,Roff,snd,snw,subl,svg,    &
                       SWout,SWsub,Tsoil,Tsrf,Tsub,Tveg,Usub,fsnow,albs)

#include "OPTS.h"

use IOUNITS, only: &

  usta                ! State output file unit number

use LAYERS, only: &
  Ncnpy,             &! Number of canopy layers
  Nsoil               ! Number of soil layers

implicit none

integer, intent(in) :: &
  Npnts,             &! Number of points
  year,              &! Year
  month,             &! Month of year
  day                 ! Day of month

real, intent(in) :: &
  hour,              &! Hour of day
  H(Npnts),          &! Sensible heat flux to the atmosphere (W/m^2)
  LE(Npnts),         &! Latent heat flux to the atmosphere (W/m^2)
  LWout(Npnts),      &! Outgoing LW radiation (W/m^2)
  LWsub(Npnts),      &! Subcanopy downward LW radiation (W/m^2)
  Melt(Npnts),       &! Surface melt rate (kg/m^2/s)
  Roff(Npnts),       &! Runoff from snow (kg/m^2/s)
  snd(Npnts),        &! Snow depth (m)
  subl(Npnts),       &! Sublimation rate (kg/m^2/s)
  svg(Npnts),        &! Total snow mass on vegetation (kg/m^2)
  snw(Npnts),        &! Total snow mass on ground (kg/m^2) 
  SWout(Npnts),      &! Outgoing SW radiation (W/m^2)
  SWsub(Npnts),      &! Subcanopy downward SW radiation (W/m^2)
  Tsoil(Nsoil,Npnts),&! Soil layer temperatures (K)
  Tsrf(Npnts),       &! Snow/ground surface temperature (K)
  Tsub(Npnts),       &! Subcanopy air temperature (K)
  Tveg(Ncnpy,Npnts), &! Vegetation layer temperatures (K)
  Usub(Npnts),       &! Subcanopy wind speed (m/s)
  fsnow(Npnts),&! Ground snowcover fraction
  albs(Npnts)   ! Snow/ground surface albedo

! State outputs
write(usta) snd,snw,Tsrf,fsnow,albs, H, LE

end subroutine FSM2_OUTPUT
